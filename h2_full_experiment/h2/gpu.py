from __future__ import annotations
import os
from .common import require_pbs

def check_gpu():
    require_pbs("workq")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selected:
        selected = os.environ.get("MIG_UUID", "").strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    if not selected.startswith("MIG-") or "," in selected:
        raise RuntimeError("Configure one currently free MIG UUID in settings.env, or use the "
                           "scheduler's MIG UUID. Numeric/full-GPU selections are not guessed.")
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(selected)
        running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        others = [p.pid for p in running if p.pid != os.getpid()]
        if others:
            raise RuntimeError("MIG instance is occupied; no training started. PIDs: " + str(others))
    finally:
        pynvml.nvmlShutdown()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one available CUDA instance")
    torch.cuda.set_device(0)
    x = torch.randn(256, 256, device="cuda")
    _ = x @ x
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    result = {"mig_uuid": selected, "name": props.name, "memory_bytes": props.total_memory,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    print("GPU COMPUTATION PASSED", result, flush=True)
    return result

