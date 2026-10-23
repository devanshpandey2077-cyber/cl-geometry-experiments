from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import uuid
import numpy as np
from .common import (ROOT, canonical_hash, data_dir, data_recipe, digest_file, load_config,
                     model_asset_path, now, read_json, require_pbs, token_dir, write_json)
from .data import (SPLITS, assign_splits, audit_splits, collect_candidates, grouped_records,
                   read_jsonl, unique_records, write_jsonl)
from .tasks import TASKS
from .prompts import PROMPT_VERSION, encode_record, label_token_ids, save_pack

def prepare_model(cfg):
    from huggingface_hub import HfApi, snapshot_download
    target = model_asset_path(cfg)
    if target.exists():
        asset = read_json(target)
        requested = cfg["model"].get("revision")
        if requested and requested != asset["revision"]:
            raise ValueError("Existing model lock differs. Use a new model asset name/project for a new revision.")
        if Path(asset["local_path"]).is_dir():
            return asset
        revision = asset["revision"]
    else:
        revision = cfg["model"].get("revision") or HfApi().model_info(cfg["model"]["repo"]).sha
    path = snapshot_download(repo_id=cfg["model"]["repo"], revision=revision, max_workers=4)
    asset = {"repo": cfg["model"]["repo"], "revision": revision,
             "local_path": str(Path(path).resolve()), "resolved_at": now()}
    write_json(target, asset)
    return asset

def prepare_data(cfg):
    target = data_dir(cfg)
    recipe_hash = canonical_hash(data_recipe(cfg))
    if (target / "manifest.json").exists():
        manifest = read_json(target / "manifest.json")
        if manifest["recipe_hash"] != recipe_hash:
            raise ValueError("Data name already used by a different recipe. Choose a new data.name.")
        for entry in manifest["files"]:
            if digest_file(target / entry["path"]) != entry["sha256"]:
                raise ValueError("Prepared data changed: " + entry["path"])
        return manifest
    if target.exists():
        raise ValueError("Existing data directory lacks a manifest; preserve/rename it before preparing.")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    revisions, candidates, source_info = {}, [], {}
    cap = max(2000, int(sum(cfg["data"]["sizes"].values()) *
                        cfg["data"]["candidate_multiplier"]))
    for task_index, name in enumerate(cfg["data"]["tasks"]):
        task = TASKS[name]
        if task.repo not in revisions:
            revisions[task.repo] = api.dataset_info(task.repo).sha
        revision = revisions[task.repo]
        dataset = load_dataset(task.repo, task.config, split="train", revision=revision)
        rows, info = collect_candidates(dataset, name, cap,
                                        np.random.default_rng(cfg["data"]["seed"] + task_index),
                                        revision, cfg["data"]["partition"])
        candidates.extend(rows)
        source_info[name] = {"repo": task.repo, "config": task.config, "revision": revision, **info}
        print("SOURCE", name, info, flush=True)
        del dataset
    candidates, unique_info = unique_records(candidates)
    print("Grouping candidate records; MinHash-LSH is approximate.", flush=True)
    groups, group_info = grouped_records(candidates, cfg["data"]["near_threshold"],
                                         cfg["data"]["minhash_permutations"],
                                         cfg["data"]["max_group_size"])
    output = assign_splits(groups, cfg["data"]["tasks"], cfg["data"]["sizes"],
                            cfg["data"]["seed"])
    audit = audit_splits(output)
    temp = target.parent / ("." + target.name + ".building_" + uuid.uuid4().hex[:8])
    temp.mkdir(parents=True)
    files = []
    for split in SPLITS:
        for task in cfg["data"]["tasks"]:
            rel = f"{task}/{split}.jsonl"
            rows = output[split][task]
            write_jsonl(temp / rel, rows)
            files.append({"path": rel, "sha256": digest_file(temp / rel), "rows": len(rows),
                          "label_counts": dict(Counter(str(r["label"]) for r in rows))})
    manifest = {"schema": 2, "created_at": now(), "recipe": data_recipe(cfg),
                "recipe_hash": recipe_hash, "sources": source_info,
                "deduplication": {**unique_info, **group_info}, "audit": audit, "files": files,
                "split_description": "custom grouped class-balanced holdouts from labeled source train splits"}
    write_json(temp / "manifest.json", manifest)
    os.replace(temp, target)
    return manifest

def prepare_tokens(cfg, asset, manifest):
    from transformers import AutoTokenizer
    target = token_dir(cfg)
    expected = {"data_manifest_hash": canonical_hash(manifest), "model": asset,
                "max_length": cfg["training"]["max_length"],
                "prompt_variant": cfg["features"]["prompt_variant"], "prompt_version": PROMPT_VERSION}
    # Ignore the resolution timestamp when comparing an existing immutable asset.
    expected["model"] = {k: asset[k] for k in ("repo", "revision", "local_path")}
    if (target / "manifest.json").exists():
        old = read_json(target / "manifest.json")
        if old["identity"] != expected:
            raise ValueError("Token cache identity mismatch")
        for entry in old["files"]:
            if digest_file(target / entry["path"]) != entry["sha256"]:
                raise ValueError("Token file changed: " + entry["path"])
        return old
    if target.exists():
        raise ValueError("Incomplete token directory; preserve/rename it before retrying.")
    tokenizer = AutoTokenizer.from_pretrained(asset["local_path"], use_fast=True,
                                               local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    codes = label_token_ids(tokenizer)
    temp = target.parent / ("." + target.name + ".building_" + uuid.uuid4().hex[:8])
    temp.mkdir(parents=True)
    files = []
    for task in cfg["data"]["tasks"]:
        for split in SPLITS:
            rows = read_jsonl(data_dir(cfg) / task / (split + ".jsonl"))
            encoded = [encode_record(tokenizer, r, cfg["training"]["max_length"],
                                      cfg["features"]["prompt_variant"]) for r in rows]
            rel = f"{task}/{split}.npz"
            save_pack(temp / rel, encoded)
            files.append({"path": rel, "sha256": digest_file(temp / rel), "rows": len(encoded),
                          "truncated_rows": sum(r["truncated"] for r in encoded),
                          "content_tokens": sum(r["content_tokens"] for r in encoded)})
            print("TOKENIZED", task, split, files[-1], flush=True)
    result = {"identity": expected, "pad_token_id": tokenizer.pad_token_id,
              "label_token_ids": codes, "label_codes": list("ABCD"), "files": files,
              "supervision": "one next-token answer-code target; no prompt-token loss",
              "mask": "tokens entirely contained in content character spans"}
    write_json(temp / "manifest.json", result)
    os.replace(temp, target)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    require_pbs("cpuq")
    cfg = load_config(args.config)
    asset = prepare_model(cfg)
    data = prepare_data(cfg)
    tokens = prepare_tokens(cfg, asset, data)
    print("PREPARATION COMPLETE", flush=True)
    print("Data:", data_dir(cfg), "\nToken cache:", token_dir(cfg),
          "\nModel:", asset["local_path"], flush=True)

if __name__ == "__main__":
    main()

