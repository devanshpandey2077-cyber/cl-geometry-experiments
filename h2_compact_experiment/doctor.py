"""Lightweight preflight: no Torch import, downloads, GPU operations or training."""
import argparse
import shutil
from h2.common import ROOT, data_dir, load_config, model_asset_path, read_json, token_dir

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/smoke.json")
args = parser.parse_args()
cfg = load_config(args.config)
paths = {"settings": ROOT / "settings.env", "model lock": model_asset_path(cfg),
         "data manifest": data_dir(cfg) / "manifest.json",
         "token manifest": token_dir(cfg) / "manifest.json"}
for name, path in paths.items():
    print(("PRESENT" if path.exists() else "MISSING"), name, path)
hidden = 2048
if model_asset_path(cfg).exists():
    asset = read_json(model_asset_path(cfg))
    config = read_json(__import__("pathlib").Path(asset["local_path"]) / "config.json")
    hidden = config["hidden_size"]
n = sum(cfg["data"]["sizes"][s] for s in cfg["features"]["probe_splits"])
feature_bytes = ((len(cfg["data"]["tasks"]) + 1) * len(cfg["data"]["tasks"]) * n *
                 len(cfg["features"]["layers"]) * len(cfg["features"]["poolings"]) * hidden * 4)
runs = len(cfg["orders"]) * len(cfg["seeds"]) + 2
print(f"Float32 feature bytes before compression: {feature_bytes / 2**30:.2f} GiB/run; "
      f"{runs * feature_bytes / 2**30:.2f} GiB for {runs} runs")
print(f"Free filesystem space: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB; "
      "check your separate college storage quota too")
print("Model cache, token/data caches, checkpoints, reports and aborted attempts are additional.")
print("CPU bootstrap analysis is substantial; it checkpoints every 25 draws and resumes on rerun.")
print("This command has not checked MIG availability. Only the PBS GPU job performs that check.")
