from __future__ import annotations
import argparse
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
from .common import (ROOT, canonical_hash, digest_file, load_config, now, read_json,
                     require_pbs, source_inventory, write_json)
from .geometry import (affinity_matrix, differences, joint_group_bootstrap_indices,
                       inversion_flags, pair_vector, simultaneous_intervals)

def load_feature(stage, probe, pooling, layer, task):
    path = Path(stage) / "features" / probe / pooling / f"layer_{layer:03d}" / (task + ".npz")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def analyze_run(run, probe=None, pooling=None, layer=None, rank=None, resamples=None):
    run = Path(run).resolve()
    if not (run / "COMPLETE.json").exists():
        raise ValueError("Run is incomplete: " + str(run))
    meta = read_json(run / "run.json")
    cfg = meta["config"]
    probe = probe or cfg["analysis"]["probe"]
    pooling = pooling or cfg["analysis"]["pooling"]
    layer = layer if layer is not None else cfg["features"]["primary_layer"]
    rank = rank if rank is not None else cfg["analysis"]["rank"]
    if rank < 1:
        raise ValueError("Rank must be positive")
    resamples = resamples if resamples is not None else cfg["analysis"]["bootstrap"]
    if resamples < 20:
        raise ValueError("Use at least 20 bootstrap draws, even for a smoke test")
    plan = {"probe": probe, "pooling": pooling, "layer": layer, "rank": rank,
            "resamples": resamples, "seed": cfg["analysis"]["bootstrap_seed"],
            "alpha": cfg["analysis"]["alpha"], "min_gap": cfg["analysis"]["min_gap"],
            "source": source_inventory()}
    out = run / "analysis" / (f"{probe}_{pooling}_l{layer}_r{rank}_" + canonical_hash(plan)[:8])
    if (out / "report.json").exists():
        old = read_json(out / "report.json")
        if old["plan"] == plan:
            return out
        raise ValueError("Analysis identity collision")
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(cfg["data"]["tasks"])
    stages = [run / "stages" / f"stage_{s:03d}" for s in range(len(tasks) + 1)]
    loaded = [{t: load_feature(stage, probe, pooling, layer, t) for t in tasks}
              for stage in stages]
    signature = {}
    for task in tasks:
        baseline = loaded[0][task]
        for rows in loaded[1:]:
            for key in ("ids", "groups", "labels"):
                if not np.array_equal(baseline[key], rows[task][key]):
                    raise ValueError("Probe identity/order changed across checkpoints")
        signature[task] = {k: baseline[k].tolist() for k in ("ids", "groups", "labels")}
    sample_signature = canonical_hash(signature)
    matrices, diagnostics, all_d = [], [], []
    for stage_index, stage_data in enumerate(loaded):
        names, matrix, diag = affinity_matrix({t: stage_data[t]["features"] for t in tasks}, rank)
        if names != tasks:
            raise AssertionError("Task order mismatch")
        matrices.append(matrix)
        diagnostics.append(diag)
        pairs, values = pair_vector(matrix)
        comparisons, diff = differences(values)
        all_d.append(diff)
        print("AFFINITY", run.name, stage_index, flush=True)
    matrices = np.asarray(matrices)
    all_d = np.asarray(all_d)
    point = all_d[[0, -1]]
    boots = np.empty((resamples, 2, len(comparisons)), dtype=np.float64)
    partial = out / "bootstrap_partial.npz"
    done = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as z:
            done = len(z["draws"])
            boots[:done] = z["draws"]
    # Same draws for every run using the same probe identities. This is required by campaign maxT.
    rng = np.random.default_rng(plan["seed"])
    for b in range(resamples):
        indices = joint_group_bootstrap_indices({t: loaded[0][t]["groups"] for t in tasks}, rng)
        if b < done:
            continue
        for endpoint, stage_index in enumerate((0, len(stages) - 1)):
            _, a, _ = affinity_matrix(
                {t: loaded[stage_index][t]["features"][indices[t]] for t in tasks}, rank)
            _, vector = pair_vector(a)
            _, boots[b, endpoint] = differences(vector)
        if b == 0 or (b + 1) % 25 == 0:
            temporary = out / "bootstrap_partial.tmp.npz"
            np.savez_compressed(temporary, draws=boots[:b+1])
            temporary.replace(partial)
            print("BOOTSTRAP", run.name, b + 1, "/", resamples, flush=True)
    lower, upper, uncertainty = simultaneous_intervals(point, boots, plan["alpha"])
    flags = inversion_flags(lower, upper, plan["min_gap"])
    raw = point[0] * point[1] < 0
    rows = []
    for k, (p, q) in enumerate(comparisons):
        a, b = pairs[p], pairs[q]
        rows.append({"comparison": k, "pair_p": "/".join(tasks[i] for i in a),
                     "pair_q": "/".join(tasks[i] for i in b), "initial_difference": float(point[0,k]),
                     "final_difference": float(point[1,k]), "initial_lower": float(lower[0,k]),
                     "initial_upper": float(upper[0,k]), "final_lower": float(lower[1,k]),
                     "final_upper": float(upper[1,k]), "raw_inversion": bool(raw[k]),
                     "within_run_interval_inversion": bool(flags[k])})
    write_csv(out / "pair_comparisons.csv", rows)
    trajectories = []
    for s, matrix in enumerate(matrices):
        for i, j in pairs:
            trajectories.append({"stage": s, "task_i": tasks[i], "task_j": tasks[j],
                                 "affinity": float(matrix[i,j])})
    write_csv(out / "affinity_trajectories.csv", trajectories)
    accuracy, accuracy_rows = [], []
    eval_split = "test" if "test" in cfg["training"]["eval_splits"] else "validation"
    for stage_index, stage in enumerate(stages):
        metrics = read_json(stage / "metrics.json")
        accuracy.append({t: metrics[t][eval_split]["accuracy"] for t in tasks})
        for task in tasks:
            accuracy_rows.append({"stage": stage_index, "task": task, "split": eval_split,
                                  **{k: metrics[task][eval_split][k]
                                     for k in ("accuracy", "macro_f1", "n", "correct",
                                               "mean_target_nll", "invalid_unrestricted_top1_rate")}})
    write_csv(out / "accuracy.csv", accuracy_rows)
    order = meta["identity"]["order"]
    final = accuracy[-1]
    bwt_values = [final[t] - accuracy[i+1][t] for i, t in enumerate(order[:-1])]
    control_drift = max(float(np.max(np.abs(loaded[s][t]["features"] - loaded[0][t]["features"])))
                        for s in range(1, len(stages)) for t in tasks)
    warnings = [
        "Bootstrap intervals are approximate and conditional on trained checkpoints.",
        "Group bootstrap targets the sampled source-group distribution; it is not a guarantee of "
        "population coverage under dependent or misspecified grouping.",
        "Within-run intervals do not adjust across runs; use campaign_summary for the joint family.",
        "Secondary layers, ranks, poolings, intermediate checkpoints and post-hoc selections "
        "require their own declared comparison family or independent confirmation.",
    ]
    if resamples < 500:
        warnings.append("Low bootstrap count: functionality/exploratory output only.")
    plot_status = "not attempted"
    try:
        from .plotting import plot_run
        plot_run(out, tasks, matrices, comparisons, pairs, all_d, lower, upper, rank, accuracy)
        plot_status = "written"
    except ImportError as e:
        plot_status = "plotting dependency unavailable: " + str(e)
    np.savez_compressed(out / "bootstrap.npz", point=point, draws=boots, lower=lower, upper=upper)
    np.savez_compressed(out / "affinities.npz", matrices=matrices, differences=all_d,
                        tasks=np.asarray(tasks, dtype=str))
    report = {"schema": 2, "created_at": now(), "run": str(run),
              "run_identity": meta["identity"], "purpose": cfg["purpose"], "plan": plan,
              "tasks": tasks, "resampling_signature": sample_signature,
              "data_recipe_hash": meta["provenance"].get("data_manifest", {}).get("recipe_hash"),
              "uncertainty": uncertainty, "raw_inversions": int(raw.sum()),
              "within_run_interval_inversions": int(flags.sum()), "comparisons": len(comparisons),
              "final_mean_accuracy": float(np.mean(list(final.values()))),
              "backward_transfer": float(np.mean(bwt_values)),
              "max_feature_change_from_baseline": control_drift,
              "fixed_representation_control_pass": (control_drift <= cfg["analysis"]["control_tolerance"]
                     if meta["identity"]["mode"] in ("frozen", "head") else None),
              "subspace_diagnostics": diagnostics, "plot_status": plot_status,
              "interpretation": "A measured rank reversal is observational evidence about this affinity "
                                "definition; it does not establish causation or novelty.",
              "warnings": warnings}
    write_json(out / "report.json", report)
    summary = [
        "# H2 run analysis", "", f"Run: {run.name}", f"Purpose: {cfg['purpose']}",
        f"Probe: {probe}; pooling: {pooling}; hidden-state index: {layer}; rank: {rank}.",
        f"Raw endpoint inversions: {int(raw.sum())}/{len(comparisons)}.",
        f"Within-run simultaneous-interval inversions: {int(flags.sum())}/{len(comparisons)}.",
        f"Final mean accuracy: {report['final_mean_accuracy']:.4f}.",
        f"Backward transfer: {report['backward_transfer']:.4f}.", "",
        "## Interpretation limits", "", *["- " + x for x in warnings], "",
        "Use the campaign report for intervals jointly adjusted across the observed runs.",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    write_json(run / "analysis" / "latest.json", {"directory": str(out), "plan": plan})
    return out

def summarize_campaign(campaign, analyses, plan=None, allow_incomplete=False):
    root = ROOT / "reports" / campaign
    root.mkdir(parents=True, exist_ok=True)
    reports = [read_json(p / "report.json") for p in analyses]
    if not reports:
        raise ValueError("No complete analyzed runs")
    first = reports[0]
    for report in reports[1:]:
        for field in ("tasks", "resampling_signature", "data_recipe_hash"):
            if report[field] != first[field]:
                raise ValueError("Joint analysis requires the same probe identities/grouping and data recipe")
        for field in ("resamples", "seed", "alpha", "min_gap", "probe", "pooling"):
            if report["plan"][field] != first["plan"][field]:
                raise ValueError("Incompatible campaign analysis settings: " + field)
    # Each draw index uses the same probe-source groups in all runs.
    points, draws = [], []
    for p in analyses:
        with np.load(p / "bootstrap.npz", allow_pickle=False) as z:
            points.append(z["point"])
            draws.append(z["draws"])
    point = np.stack(points)
    boot = np.stack(draws, axis=1)
    lower, upper, uncertainty = simultaneous_intervals(point, boot, first["plan"]["alpha"])
    counts, detail = [], []
    controls = []
    pairs = list(combinations(first["tasks"], 2))
    comparison_names = list(combinations(pairs, 2))
    for i, report in enumerate(reports):
        flags = inversion_flags(lower[i], upper[i], first["plan"]["min_gap"])
        identity = report["run_identity"]
        counts.append({"run": Path(report["run"]).name, "model": Path(report["run"]).parent.name,
                       "mode": identity["mode"], "order": identity["order_name"], "seed": identity["seed"],
                       "joint_interval_inversions": int(flags.sum()),
                       "raw_inversions": report["raw_inversions"],
                       "final_mean_accuracy": report["final_mean_accuracy"],
                       "backward_transfer": report["backward_transfer"],
                       "control_pass": report["fixed_representation_control_pass"]})
        if identity["mode"] in ("frozen", "head"):
            controls.append(report["fixed_representation_control_pass"])
        for k in range(point.shape[-1]):
            detail.append({"run": Path(report["run"]).name, "model": Path(report["run"]).parent.name,
                           "comparison": k, "pair_p": "/".join(comparison_names[k][0]),
                           "pair_q": "/".join(comparison_names[k][1]),
                           "initial_difference": float(point[i,0,k]),
                           "final_difference": float(point[i,1,k]),
                           "initial_lower": float(lower[i,0,k]), "initial_upper": float(upper[i,0,k]),
                           "final_lower": float(lower[i,1,k]), "final_upper": float(upper[i,1,k]),
                           "joint_interval_inversion": bool(flags[k])})
    write_csv(root / "runs.csv", counts)
    write_csv(root / "joint_pair_comparisons.csv", detail)
    # Resample training seeds as blocks across task orders, not pair comparisons as independent units.
    seed_variation = {}
    for model in sorted({r["model"] for r in counts}):
        rows = [r for r in counts if r["model"] == model and r["mode"] == "lora"]
        seeds = sorted({r["seed"] for r in rows})
        if not seeds:
            continue
        seed_means = np.asarray([np.mean([r["joint_interval_inversions"] for r in rows if r["seed"] == s])
                                 for s in seeds])
        rng = np.random.default_rng(77119)
        means = rng.choice(seed_means, size=(10000, len(seeds)), replace=True).mean(axis=1)
        seed_variation[model] = {"independent_seeds": len(seeds),
             "mean_joint_inversion_count": float(seed_means.mean()),
             "descriptive_seed_bootstrap_95_interval": np.quantile(means, [.025,.975]).tolist(),
             "scope": "descriptive stability across observed training seeds, conditional on this data "
                      "and the joint probe-interval decision rule; not a new hypothesis test"}
    complete = not allow_incomplete
    record = {"created_at": now(), "campaign": campaign, "complete_plan": complete,
              "runs": counts, "joint_uncertainty": uncertainty,
              "seed_variation": seed_variation,
              "controls_present": len(controls), "all_present_controls_pass": all(controls) if controls else False,
              "plan": plan, "conclusion": "Inspect joint intervals, independent-run stability, controls, "
              "and scope. This program does not declare H2 proven, establish causality, or assess novelty."}
    write_json(root / "campaign_report.json", record)
    lines = ["# H2 campaign evidence summary", "", f"Campaign: {campaign}",
             f"Observed complete runs: {len(reports)}", f"Plan complete: {complete}",
             f"Joint endpoint interval family size: {uncertainty['family_size']}",
             "Probe intervals are approximate, jointly adjusted across the supplied runs and "
             "conditional on those trained checkpoints.", "",
             "| Model | Run | Mode | Joint inversions | Final accuracy |",
             "|---|---|---|---:|---:|"]
    for r in counts:
        lines.append(f"| {r['model']} | {r['run']} | {r['mode']} | "
                     f"{r['joint_interval_inversions']} | {r['final_mean_accuracy']:.3f} |")
    lines += ["", "## Limits", "",
              "- Independent training variability is summarized by resampling seeds as blocks.",
              "- Probe sampling uncertainty and training-seed variability are distinct.",
              "- Failed/missing controls and incomplete plans prevent a confirmatory interpretation.",
              "- A successful observational test does not establish H3, H4, or novelty.",
              "- Full subspace diagnostics, comparison rows and raw data accompany this summary."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--campaign")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--probe", choices=["probe_a","probe_b"])
    parser.add_argument("--pooling", choices=["content","answer","prompt","last_content"])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.campaign and not args.campaign.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid campaign name")
    require_pbs("cpuq")
    if bool(args.run) == bool(args.campaign):
        raise ValueError("Choose either --run or --campaign")
    if args.run:
        out = analyze_run(args.run, args.probe, args.pooling, args.layer, args.rank, args.bootstrap)
    else:
        base = ROOT / "runs" / args.campaign
        plan_file = ROOT / "campaigns" / (args.campaign + ".json")
        plan = read_json(plan_file) if plan_file.exists() else None
        if not plan and not args.allow_incomplete:
            raise ValueError("No registered campaign plan; use --allow-incomplete for a descriptive summary")
        run_paths = [ROOT / row["run_path"] for row in plan["runs"]] if plan else sorted(
            p.parent for p in base.glob("*/*/run.json"))
        missing = [str(p) for p in run_paths if not (p / "COMPLETE.json").exists()]
        if missing and not args.allow_incomplete:
            raise ValueError("Campaign incomplete. Missing/failed runs: " + str(missing))
        analyses = [analyze_run(p, args.probe, args.pooling, args.layer, args.rank, args.bootstrap)
                    for p in run_paths if (p / "COMPLETE.json").exists()]
        out = summarize_campaign(args.campaign, analyses, plan, args.allow_incomplete)
    print("ANALYSIS COMPLETE", out, flush=True)

if __name__ == "__main__":
    main()

