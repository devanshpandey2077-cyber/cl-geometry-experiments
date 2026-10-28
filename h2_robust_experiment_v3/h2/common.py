from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_hash(obj):
    return digest_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out

def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if "extends" in cfg:
        parent = cfg.pop("extends")
        cfg = merge(load_config(path.parent / parent), cfg)
    required = ("name", "data", "model", "training", "features", "analysis", "orders", "seeds")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError("Missing configuration fields: " + str(missing))
    tasks = cfg["data"]["tasks"]
    for key, order in cfg["orders"].items():
        if sorted(order) != sorted(tasks):
            raise ValueError("Order must contain every task exactly once: " + key)
    if not 0 < cfg["analysis"]["alpha"] < 1:
        raise ValueError("alpha must be between zero and one")
    if cfg["features"]["primary_layer"] not in cfg["features"]["layers"]:
        raise ValueError("Primary layer must be extracted")
    if len(set(tasks)) != len(tasks) or len(tasks)<3:
        raise ValueError("Need >=3 unique tasks")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]): raise ValueError("Repeated seed")
    for key in ("microbatch","gradient_accumulation","eval_batch","epochs","max_length"):
        if not isinstance(cfg["training"][key],int) or cfg["training"][key]<1:
            raise ValueError("Positive integer required: "+key)
    if cfg["training"]["learning_rate"]<=0: raise ValueError("Learning rate must be positive")
    if cfg["analysis"]["rank"] >= min(cfg["data"]["sizes"][s] for s in cfg["features"]["probe_splits"])-1:
        raise ValueError("Probe size too small for requested analysis rank")
    return cfg

def data_recipe(cfg):
    return cfg["data"]

def model_slug(cfg):
    return cfg["model"]["repo"].replace("/", "--")

def data_dir(cfg):
    return ROOT / "data" / cfg["data"]["name"]

def token_dir(cfg):
    key = canonical_hash({"data": data_recipe(cfg), "repo": cfg["model"]["repo"],
                          "revision": cfg["model"].get("revision"),
                          "max_length": cfg["training"]["max_length"],
                          "prompt_variant": cfg["features"]["prompt_variant"]})[:16]
    return ROOT / "data" / "tokens" / model_slug(cfg) / key

def model_asset_path(cfg):
    return ROOT / "assets" / (model_slug(cfg) + ".json")

def source_inventory():
    paths = sorted((ROOT / "h2").glob("*.py")) + sorted((ROOT / "pbs").glob("*"))
    paths += [ROOT / "requirements.txt"]
    paths += sorted(ROOT.glob("*.py"))
    return {str(p.relative_to(ROOT)): digest_file(p) for p in paths if p.is_file()}

def software_inventory():
    packages = ("torch", "transformers", "datasets", "huggingface-hub", "numpy",
                "scipy", "nvidia-ml-py")
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out

def require_pbs(queue):
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Submit through PBS. This command does not run on the login shell.")
    actual = os.environ.get("PBS_QUEUE") or os.environ.get("PBS_O_QUEUE")
    if actual and actual.split("@")[0] != queue:
        raise RuntimeError("Wrong PBS queue: expected " + queue + ", got " + actual)

def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_numpy(tensor):
    import numpy as np
    tensor = tensor.detach().float().cpu()
    try:
        return tensor.numpy()
    except RuntimeError as e:
        if "Numpy" not in str(e) and "NumPy" not in str(e):
            raise
        return np.asarray(tensor.tolist(), dtype=np.float32)

def as_tensor(array, dtype=None, device=None):
    import torch
    try:
        return torch.as_tensor(array, dtype=dtype, device=device)
    except (RuntimeError, TypeError) as e:
        if hasattr(array, "tolist"):
            return torch.tensor(array.tolist(), dtype=dtype, device=device)
        raise e

