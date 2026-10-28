from __future__ import annotations
import argparse
from collections import Counter, defaultdict
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
from .robust_data import stable_assign

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
    path = snapshot_download(repo_id=cfg["model"]["repo"], revision=revision, max_workers=4,
        allow_patterns=["*.safetensors","*.json","*.model","*.txt","*.tiktoken"])
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    files = [{"path":p.relative_to(path).as_posix(),"sha256":digest_file(p),"bytes":p.stat().st_size}
             for p in sorted(Path(path).rglob("*")) if p.is_file()]
    asset = {"repo": cfg["model"]["repo"], "revision": revision,
             "local_path": str(Path(path).resolve()), "resolved_at": now(),"files":files,
             "hidden_size":model_config.hidden_size,"num_hidden_layers":model_config.num_hidden_layers}
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
    lock_path = ROOT / "assets" / "dataset_revisions.json"
    revisions = read_json(lock_path) if lock_path.exists() else {}
    candidates, source_info = [], {}
    cap = cfg["data"].get("candidate_cap") or max(2000, int(sum(cfg["data"]["sizes"].values()) *
                        cfg["data"]["candidate_multiplier"]))
    pool_recipe = {k:cfg["data"].get(k) for k in ("tasks","partition","candidate_seed",
                    "candidate_cap","near_threshold","minhash_permutations","max_group_size")}
    # Lock source revisions before identifying a shared pool.
    for name in cfg["data"]["tasks"]:
        repo=TASKS[name].repo
        if repo not in revisions: revisions[repo]=api.dataset_info(repo).sha
    write_json(lock_path,revisions)
    pool_recipe["source_revisions"]=revisions
    pool_recipe["data_implementation"]={n:digest_file(ROOT/"h2"/n) for n in ("data.py","robust_data.py")}
    pool = ROOT / "data" / "group_pools" / canonical_hash(pool_recipe)[:20]
    cached = (pool/"manifest.json").exists()
    if cached:
        pool_meta=read_json(pool/"manifest.json")
        if digest_file(pool/"rows.jsonl") != pool_meta["sha256"]: raise ValueError("Group pool changed")
        buckets=defaultdict(list)
        for row in read_jsonl(pool/"rows.jsonl"): buckets[row["group_id"]].append(row)
        groups=list(buckets.values())
        source_info=pool_meta["sources"]
        unique_info,group_info=pool_meta["unique"],pool_meta["groups"]
    for task_index, name in enumerate(cfg["data"]["tasks"]):
        if cached: break
        task = TASKS[name]
        if task.repo not in revisions:
            revisions[task.repo] = api.dataset_info(task.repo).sha
        revision = revisions[task.repo]
        dataset = load_dataset(task.repo, task.config, split="train", revision=revision)
        rows, info = collect_candidates(dataset, name, cap,
                                        np.random.default_rng(cfg["data"].get("candidate_seed",cfg["data"]["seed"]) + task_index),
                                        revision, cfg["data"]["partition"])
        candidates.extend(rows)
        source_info[name] = {"repo": task.repo, "config": task.config, "revision": revision, **info}
        print("SOURCE", name, info, flush=True)
        del dataset
    if not cached:
        candidates, unique_info = unique_records(candidates)
        print("Grouping candidate records; MinHash-LSH is approximate.", flush=True)
        groups, group_info = grouped_records(candidates, cfg["data"]["near_threshold"],
                                         cfg["data"]["minhash_permutations"],
                                         cfg["data"]["max_group_size"])
        write_jsonl(pool/"rows.jsonl",[r for g in groups for r in g])
        write_json(pool/"manifest.json",{"recipe":pool_recipe,"sources":source_info,
            "sha256":digest_file(pool/"rows.jsonl"),"unique":unique_info,"groups":group_info})
    if "stable_split_seed" in cfg["data"]:
        output=stable_assign(groups,cfg["data"]["tasks"],cfg["data"]["sizes"],
                             cfg["data"]["seed"],cfg["data"]["stable_split_seed"])
    else:
        output = assign_splits(groups, cfg["data"]["tasks"], cfg["data"]["sizes"],cfg["data"]["seed"])
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
                "group_pool":pool.relative_to(ROOT).as_posix(),
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

