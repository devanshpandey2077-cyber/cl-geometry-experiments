"""Read saved run state without importing Torch or starting computation."""
import argparse
from pathlib import Path
from h2.common import ROOT, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--campaign", required=True)
args = parser.parse_args()
if not args.campaign.replace("_", "").replace("-", "").isalnum():
    parser.error("Invalid campaign name")
plan = read_json(ROOT / "campaigns" / (args.campaign + ".json"))
for row in plan["runs"]:
    run = ROOT / row["run_path"]
    state = ("COMPLETE" if (run / "COMPLETE.json").exists() else
             "FAILED/INTERRUPTED" if (run / "FAILURE.json").exists() else
             "STARTED" if (run / "run.json").exists() else "NOT STARTED")
    print(f"{state:20s} {row.get('job_id', '?'):16s} {row['run_path']}")
print("PBS state: qstat -u YOUR_HPC_USER")
print("Current/recent GPU log: tail -n 60 logs/gpu.latest.log")
print("Analysis log: tail -n 60 logs/analysis.latest.log")
print("Results:", ROOT / "reports" / args.campaign / "summary.md")
