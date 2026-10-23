from __future__ import annotations
import argparse
from contextlib import nullcontext
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid
import numpy as np
import torch
from torch import nn
from .common import (ROOT, canonical_hash, data_dir, digest_file, load_config, model_asset_path,
                     model_slug, now, read_json, seed_everything, software_inventory,
                     source_inventory, to_numpy, token_dir, write_json)
from .lora import inject_lora, restore_trainable, trainable_state
from .prompts import Pack, collate, pool_hidden
from .tasks import TASKS

def forward_model(model, batch, hidden=False):
    kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids")}
    kwargs.update(output_hidden_states=hidden, use_cache=False, return_dict=True)
    signature = inspect.signature(model.forward).parameters
    if "logits_to_keep" in signature:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in signature:
        kwargs["num_logits_to_keep"] = 1
    return model(**kwargs)

def answer_logits(model, output, batch, mode, classes, codes):
    if mode == "head":
        hidden = pool_hidden(output.hidden_states[-1], batch, "content").detach()
        return model.h2_readout(hidden.float())[:, :classes]
    return output.logits[:, -1, :]

def loss_and_scores(model, output, batch, mode, classes, codes):
    logits = answer_logits(model, output, batch, mode, classes, codes).float()
    if mode == "head":
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="sum")
        restricted = logits
        invalid = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
    else:
        ids = torch.tensor(codes[:classes], device=logits.device)
        targets = ids[batch["labels"]]
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        restricted = logits[:, ids]
        invalid = ~(logits.argmax(-1)[:, None] == ids[None, :]).any(-1)
    return loss, restricted, invalid

def evaluate(model, pack, classes, codes, pad_id, device, batch_size, mode):
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    total_loss, invalid_count, count = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(pack), batch_size):
            rows = [pack.row(i) for i in range(start, min(start + batch_size, len(pack)))]
            batch = collate(rows, pad_id, device)
            out = forward_model(model, batch, hidden=(mode == "head"))
            loss, scores, invalid = loss_and_scores(model, out, batch, mode, classes, codes)
            pred = scores.argmax(-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            for y, p in zip(labels, pred):
                confusion[y, p] += 1
            total_loss += float(loss)
            invalid_count += int(invalid.sum())
            count += len(rows)
    if count == 0:
        raise ValueError("Empty evaluation set")
    f1 = []
    for c in range(classes):
        tp = int(confusion[c, c])
        denom = int(confusion[c].sum() + confusion[:, c].sum())
        f1.append(2 * tp / denom if denom else 0.0)
    return {"n": count, "correct": int(np.trace(confusion)),
            "accuracy": float(np.trace(confusion) / count), "macro_f1": float(np.mean(f1)),
            "mean_target_nll": total_loss / count,
            "invalid_unrestricted_top1_rate": invalid_count / count,
            "confusion": confusion.tolist()}

def extract(model, pack, layers, poolings, pad_id, device, batch_size, destination, task):
    model.eval()
    arrays = {(layer, pooling): [] for layer in layers for pooling in poolings}
    with torch.inference_mode():
        for start in range(0, len(pack), batch_size):
            rows = [pack.row(i) for i in range(start, min(start + batch_size, len(pack)))]
            batch = collate(rows, pad_id, device)
            out = forward_model(model, batch, hidden=True)
            for layer in layers:
                if layer < 1 or layer >= len(out.hidden_states):
                    raise ValueError("Hidden-state index out of range: " + str(layer))
                for pooling in poolings:
                    arrays[layer, pooling].append(to_numpy(pool_hidden(out.hidden_states[layer],
                                                                       batch, pooling)))
    for (layer, pooling), chunks in arrays.items():
        path = Path(destination) / pooling / f"layer_{layer:03d}" / (task + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        x = np.concatenate(chunks).astype(np.float32)
        if len(x) != len(pack) or not np.isfinite(x).all():
            raise ValueError("Invalid extracted features")
        np.savez_compressed(path, features=x, ids=pack.arrays["ids"],
                            groups=pack.arrays["groups"], labels=pack.arrays["labels"])

def train_task(model, pack, cfg, mode, codes, pad_id, device, seed, task, event_path):
    if mode == "frozen":
        return {"updates": 0, "examples": 0, "input_tokens": 0}
    settings = cfg["training"]
    micro, accum = settings["microbatch"], settings["gradient_accumulation"]
    n_batches = math.ceil(len(pack) / micro)
    total_steps = math.ceil(n_batches / accum) * settings["epochs"]
    cap = settings["max_steps_per_task"]
    total_steps = min(total_steps, cap) if cap is not None else total_steps
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    warmup = max(1, int(total_steps * settings["warmup_fraction"]))
    updates, examples, tokens, loss_total = 0, 0, 0, 0.0
    start_time = time.monotonic()
    for epoch in range(settings["epochs"]):
        permutation = np.random.default_rng(seed + epoch).permutation(len(pack))
        batches = [permutation[i:i+micro] for i in range(0, len(pack), micro)]
        for begin in range(0, len(batches), accum):
            if updates >= total_steps:
                break
            group = batches[begin:begin+accum]
            group_size = sum(len(indices) for indices in group)
            if mode == "head":
                model.eval()  # The feature-producing backbone stays deterministic and frozen.
                model.h2_readout.train()
            else:
                model.train()
            optimizer.zero_grad(set_to_none=True)
            warm = min(1.0, (updates + 1) / warmup)
            progress = max(0, updates - warmup) / max(1, total_steps - warmup)
            lr = settings["learning_rate"] * warm * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            group_loss = 0.0
            for indices in group:
                rows = [pack.row(int(i)) for i in indices]
                batch = collate(rows, pad_id, device)
                context = torch.no_grad() if mode == "head" else nullcontext()
                with context:
                    out = forward_model(model, batch, hidden=(mode == "head"))
                loss, _, _ = loss_and_scores(model, out, batch, mode, len(TASKS[task].labels), codes)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / group_size).backward()
                group_loss += float(loss.detach())
                examples += len(rows)
                tokens += int(batch["attention_mask"].sum())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"]))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            updates += 1
            loss_total += group_loss
            event = {"at": now(), "task": task, "epoch": epoch, "update": updates,
                     "loss": group_loss / group_size, "lr": lr, "grad_norm": grad_norm,
                     "examples_seen": examples, "input_tokens_seen": tokens}
            with Path(event_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            if updates == 1 or updates % 25 == 0:
                print("TRAIN", event, flush=True)
    model.eval()
    return {"updates": updates, "examples": examples, "input_tokens": tokens,
            "mean_target_nll_training": loss_total / max(1, examples),
            "seconds": time.monotonic() - start_time}

def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def build_model(cfg, asset, mode, device):
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, cfg["model"]["precision"]) if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(asset["local_path"], dtype=dtype,
                 local_files_only=True, trust_remote_code=False, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    model.requires_grad_(False)
    matched = []
    if mode == "lora":
        matched = inject_lora(model, cfg["training"]["lora"])
        if cfg["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    elif mode == "head":
        model.h2_readout = nn.Linear(model.config.hidden_size, 4, device=device,
                                    dtype=torch.float32)
    elif mode != "frozen":
        raise ValueError("Unknown mode")
    return model, matched

def execute_run(cfg, model, packs, token_manifest, run_path, mode, order_name, seed,
                device, resume=False, provenance=None):
    """Shared production/test engine. Resume is only from complete task boundaries."""
    run_path = Path(run_path)
    config_hash = canonical_hash(cfg)
    identity = {"config_hash": config_hash, "mode": mode, "order_name": order_name,
                "order": cfg["orders"][order_name], "seed": seed,
                "token_manifest_hash": canonical_hash(token_manifest)}
    sources = source_inventory()
    completed = []
    if run_path.exists():
        if not resume:
            raise FileExistsError("Run already exists; use a new campaign or explicit --resume")
        old = read_json(run_path / "run.json")
        if old["identity"] != identity or old["source_sha256"] != sources:
            raise ValueError("Resume configuration/data/code differs from the original run")
        for index in range(len(identity["order"]) + 1):
            stage = run_path / "stages" / f"stage_{index:03d}"
            if (stage / "COMPLETE.json").exists():
                completed.append(index)
            else:
                break
        if completed:
            saved = torch.load(run_path / "stages" / f"stage_{completed[-1]:03d}" /
                               "checkpoint.pt", map_location="cpu", weights_only=True)
            restore_trainable(model, saved["parameters"])
            restore_rng(saved["rng"])
        if (run_path / "COMPLETE.json").exists():
            print("RUN ALREADY COMPLETE", run_path, flush=True)
            return
    else:
        run_path.mkdir(parents=True)
        for name in sources:
            dst = run_path / "code" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dst)
        write_json(run_path / "run.json", {"created_at": now(), "identity": identity,
                   "config": cfg, "source_sha256": sources, "software": software_inventory(),
                   "provenance": provenance or {}, "notes":
                   "Custom LoRA checkpoint format; optimizer reset at each task boundary."})
    stages_root = run_path / "stages"
    stages_root.mkdir(exist_ok=True)
    for stale in stages_root.glob("*.tmp"):
        archived = run_path / "aborted"
        archived.mkdir(exist_ok=True)
        stale.rename(archived / (stale.name + "_" + uuid.uuid4().hex[:8]))
    start_time = time.monotonic()
    try:
        for stage_index in range((completed[-1] + 1) if completed else 0, len(identity["order"]) + 1):
            stage = stages_root / f"stage_{stage_index:03d}"
            temp = stages_root / (stage.name + ".tmp")
            temp.mkdir()
            trained = identity["order"][stage_index - 1] if stage_index else None
            training = None
            if trained:
                training = train_task(model, packs[trained]["train"], cfg, mode,
                                      token_manifest["label_token_ids"],
                                      token_manifest["pad_token_id"], device,
                                      seed + stage_index * 100003, trained,
                                      run_path / "training_events.jsonl")
            metrics = {}
            for task in cfg["data"]["tasks"]:
                metrics[task] = {}
                for split in cfg["training"]["eval_splits"]:
                    metrics[task][split] = evaluate(model, packs[task][split],
                        len(TASKS[task].labels), token_manifest["label_token_ids"],
                        token_manifest["pad_token_id"], device,
                        cfg["training"]["eval_batch"], mode)
                for split in cfg["features"]["probe_splits"]:
                    extract(model, packs[task][split], cfg["features"]["layers"],
                            cfg["features"]["poolings"], token_manifest["pad_token_id"],
                            device, cfg["training"]["eval_batch"], temp / "features" / split, task)
                    print("PROBED", stage_index, task, split, flush=True)
            write_json(temp / "metrics.json", metrics)
            write_json(temp / "training.json", training)
            torch.save({"parameters": trainable_state(model), "rng": rng_state()},
                       temp / "checkpoint.pt")
            write_json(temp / "COMPLETE.json", {"stage": stage_index, "task": trained,
                       "mode": mode, "at": now(), "feature_layer_indexing":
                       "hidden_states[0] is embedding output; index k is after k decoder blocks "
                       "(zero-based block k-1); final index may include the final model norm"})
            os.replace(temp, stage)
            print("STAGE COMPLETE", stage_index, metrics, flush=True)
        memory = {}
        if device == "cuda":
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write_json(run_path / "COMPLETE.json", {"finished_at": now(),
                   "seconds_this_invocation": time.monotonic() - start_time, **memory})
        print("TRAINING COMPLETE", run_path, flush=True)
    except BaseException as error:
        write_json(run_path / "FAILURE.json", {"at": now(), "type": type(error).__name__,
                                             "message": str(error)})
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--mode", choices=["lora", "frozen", "head"], default="lora")
    parser.add_argument("--order", default="O1")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--campaign", default="first_smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.campaign.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Campaign must contain only letters, numbers, underscores and hyphens")
    from .gpu import check_gpu
    hardware = check_gpu()
    cfg = load_config(args.config)
    if args.order not in cfg["orders"]:
        raise ValueError("Unknown order")
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()
    asset = read_json(model_asset_path(cfg))
    data = read_json(data_dir(cfg) / "manifest.json")
    tokens = read_json(token_dir(cfg) / "manifest.json")
    if tokens["identity"]["data_manifest_hash"] != canonical_hash(data):
        raise ValueError("Data/token identity mismatch; prepare this configuration first")
    if tokens["identity"]["model"]["revision"] != asset["revision"]:
        raise ValueError("Model/token revision mismatch")
    for entry in tokens["files"]:
        if digest_file(token_dir(cfg) / entry["path"]) != entry["sha256"]:
            raise ValueError("Token cache modified: " + entry["path"])
    packs = {t: {s: Pack(token_dir(cfg) / t / (s + ".npz"))
                 for s in ("train", "validation", "probe_a", "probe_b", "test")}
             for t in cfg["data"]["tasks"]}
    model, modules = build_model(cfg, asset, args.mode, "cuda")
    run = ROOT / "runs" / args.campaign / model_slug(cfg) / f"{args.mode}_{args.order}_s{args.seed}"
    provenance = {"pbs_job_id": os.environ["PBS_JOBID"], "host": os.uname().nodename,
                  "model_asset": asset, "data_manifest": data, "hardware": hardware,
                  "lora_modules": modules,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    execute_run(cfg, model, packs, tokens, run, args.mode, args.order, args.seed, "cuda",
                args.resume, provenance)

if __name__ == "__main__":
    main()

