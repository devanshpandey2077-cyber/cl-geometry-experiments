"""Plan or submit one serial GPU campaign. No training runs in this process."""
from __future__ import annotations
import argparse
import getpass
from pathlib import Path
import re
import shlex
import subprocess
from h2.common import ROOT, canonical_hash, load_config, model_slug, now, read_json, write_json


def campaign_rows(cfg, campaign, orders, seeds, controls=True):
    rows = []
    jobs = [("lora", o, s) for s in seeds for o in orders]
    if controls:
        jobs += [(mode, orders[0], seeds[0]) for mode in ("frozen", "head")]
    for mode, order, seed in jobs:
        rows.append({"mode": mode, "order": order, "seed": seed,
                     "run_path": (Path("runs") / campaign / model_slug(cfg) /
                                  f"{mode}_{order}_s{seed}").as_posix()})
    return rows


def pbs_variables(values):
    for key, value in values.items():
        if any(c in str(value) for c in (",", "\n", "\r")):
            raise ValueError("Invalid PBS variable: " + key)
    return ",".join(f"{k}={v}" for k, v in values.items())


def checked_submit(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("qsub failed: " + result.stderr.strip())
    job = result.stdout.strip()
    if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?", job):
        raise RuntimeError("Unrecognized qsub reply. Inspect qstat before retrying: " + job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--orders", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--after", help="An existing PBS job to depend on")
    parser.add_argument("--with-analysis", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Without this flag, print the plan only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign):
        parser.error("Campaign names contain only letters, digits, underscore and hyphen")
    if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", args.walltime):
        parser.error("Walltime format is HH:MM:SS")
    if args.after and not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?", args.after):
        parser.error("Invalid --after job ID")
    config_path = Path(args.config).resolve()
    if not config_path.is_relative_to(ROOT):
        parser.error("Keep the configuration inside this project")
    relative_config = config_path.relative_to(ROOT).as_posix()
    cfg = load_config(config_path)
    orders, seeds = args.orders or list(cfg["orders"]), args.seeds or cfg["seeds"]
    if not seeds or not orders or any(o not in cfg["orders"] for o in orders):
        parser.error("Empty seeds/orders, or an order not present in the configuration")
    if len(set(seeds)) != len(seeds) or len(set(orders)) != len(orders):
        parser.error("Do not repeat a seed or order")
    rows = campaign_rows(cfg, args.campaign, orders, seeds, not args.no_controls)
    identity = canonical_hash({"config": cfg, "runs": rows})
    path = ROOT / "campaigns" / (args.campaign + ".json")
    old = read_json(path) if path.exists() else None
    if old and old["identity"] != identity:
        raise ValueError("This campaign name already has a different plan; choose a new name")
    if old and args.submit and not args.resume:
        raise ValueError("Campaign already registered. Inspect status; use --resume after clearing old pending jobs")
    if args.submit:
        if not (ROOT / "settings.env").exists():
            raise FileNotFoundError("Copy settings.env.example to settings.env and set the current MIG UUID")
        result = subprocess.run(["qstat", "-u", getpass.getuser()], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Could not inspect your PBS jobs; inspect qstat before submitting")
        active = [line for line in result.stdout.splitlines()
                  if "workq" in line.split() and "C" not in line.split()]
        if active and not args.after:
            raise RuntimeError("You already have workq jobs. Inspect qstat; use --after for an intentional dependency:\n"
                               + "\n".join(active))
    plan = old or {"schema": 2, "created_at": now(), "identity": identity,
                   "config_path": relative_config, "resolved_config": cfg, "runs": rows}
    # Write the ledger after every accepted submission, before attempting the next one.
    if args.submit:
        write_json(path, plan)
    previous = args.after
    for row in plan["runs"]:
        run = ROOT / row["run_path"]
        if (run / "COMPLETE.json").exists():
            print("SKIP COMPLETE", row["run_path"])
            continue
        values = {"H2_CONFIG": relative_config, "H2_CAMPAIGN": args.campaign,
                  "H2_MODE": row["mode"], "H2_ORDER": row["order"],
                  "H2_SEED": row["seed"], "H2_RESUME": int(args.resume)}
        command = ["qsub", "-l", "walltime=" + args.walltime, "-v", pbs_variables(values)]
        if previous:
            command += ["-W", "depend=afterok:" + previous]
        command += ["pbs/gpu.pbs"]
        print(shlex.join(command))
        if args.submit:
            job = checked_submit(command)
            row.setdefault("submission_history", []).append({"at": now(), "job_id": job})
            row["job_id"] = job
            write_json(path, plan)
            previous = job
            print("ACCEPTED", job, row["run_path"], flush=True)
    if args.with_analysis:
        command = ["qsub", "-v", pbs_variables({"H2_CAMPAIGN": args.campaign})]
        if previous:
            command += ["-W", "depend=afterok:" + previous]
        command += ["pbs/analyze.pbs"]
        print(shlex.join(command))
        if args.submit:
            plan["analysis_job_id"] = checked_submit(command)
            write_json(path, plan)
            print("ANALYSIS JOB", plan["analysis_job_id"])
    print("SUBMITTED" if args.submit else "DRY RUN: no jobs submitted and no plan written")
    print("Campaign:", args.campaign, "; GPU runs:", len(rows))


if __name__ == "__main__":
    main()
