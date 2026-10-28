"""Atomic trusted-local checkpoints, cooperative pause, and process locks."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import time

STOP = False
START = time.monotonic()

class RunPaused(RuntimeError):
    pass

def install_handlers():
    def handler(signum, frame):
        global STOP
        STOP = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)

def pause_requested():
    minutes = float(os.environ.get("H2_MAX_MINUTES", "0"))
    return STOP or (minutes > 0 and time.monotonic() - START > minutes * 60)

def atomic_save(value, path):
    import torch
    from .storage import guard,tensor_bytes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    guard(path.parent, tensor_bytes(value) + 16*2**20)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

@contextmanager
def exclusive_lock(path,blocking=False):
    """OS releases this lock on process exit; a leftover filename is harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
