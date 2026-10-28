"""Disk guards. These are headroom checks, not reservations on a shared filesystem."""
import os
import shutil
from pathlib import Path

GIB = 2**30


def guard(path, additional=0, reserve_gib=None):
    value = os.environ.get("H2_DISK_RESERVE_GIB") if reserve_gib is None else reserve_gib
    if value is None:
        return
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    required = int(float(value) * GIB) + int(additional)
    if free < required:
        raise RuntimeError(f"Insufficient disk headroom: {free/GIB:.2f} GiB free, "
                           f"{required/GIB:.2f} GiB required at {path}. "
                           "Preserve the run; free approved space and resume. No files were deleted.")


def tensor_bytes(value):
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def feature_bytes(cell):
    cfg = cell["config"]
    from suite import MODELS
    k = len(cfg["data"]["tasks"])
    hidden = MODELS[cell["model_key"]]["hidden"]
    probes = sum(cfg["data"]["sizes"][s] for s in cfg["features"]["probe_splits"])
    endpoints = 2*k*probes*hidden*4*len(cfg["features"]["layers"])*len(cfg["features"]["poolings"])
    middle = (k-1)*k*cfg["data"]["sizes"][cfg["analysis"]["probe"]]*hidden*4
    return endpoints + middle


def check_run_space(plan, row):
    from .common import ROOT
    cell = next(c for c in plan["cells"] if c["id"] == row["cell"])
    # Conservative working allowance in addition to feature payload and the protected reserve.
    # Actual archive sizes are recorded by pack; interrupted/aborted files already reduce free space.
    extra = feature_bytes(cell) * 1.25 + 1.5*GIB
    guard(ROOT, extra, plan["storage"]["reserve_gib"])
    return {"free_gib": shutil.disk_usage(ROOT).free/GIB,
            "required_gib": extra/GIB + plan["storage"]["reserve_gib"]}
