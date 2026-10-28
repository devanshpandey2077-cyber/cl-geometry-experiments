"""Stable split ownership across repeated data sampling, with explicit overlap audits."""
from collections import Counter
import hashlib
import numpy as np
from .data import SPLITS, quota, audit_splits
from .tasks import TASKS

def group_role(group_id, split_seed):
    value = int.from_bytes(hashlib.sha256(f"{split_seed}:{group_id}".encode()).digest()[:8], "big") % 10000
    # Fixed independently of requested quotas and sampling seed.
    for ceiling, role in ((7000,"train"),(7500,"validation"),(8300,"probe_a"),
                          (9100,"probe_b"),(10000,"test")):
        if value < ceiling:
            return role
    raise AssertionError("Unreachable")

def stable_assign(groups, task_names, sizes, sample_seed, split_seed):
    remaining = {s: {(t,c): n for t in task_names
                    for c,n in enumerate(quota(int(sizes[s]),len(TASKS[t].labels)))} for s in SPLITS}
    result = {s: {t: [] for t in task_names} for s in SPLITS}
    for index in np.random.default_rng(sample_seed).permutation(len(groups)):
        group = groups[int(index)]
        role = group_role(group[0]["group_id"],split_seed)
        counts = Counter((r["task"],r["label"]) for r in group)
        if all(remaining[role][key] >= count for key,count in counts.items()):
            for row in group: result[role][row["task"]].append(row)
            for key,count in counts.items(): remaining[role][key] -= count
    deficits = {s: {f"{t}:{c}": n for (t,c),n in values.items() if n}
                for s,values in remaining.items() if any(values.values())}
    if deficits:
        raise ValueError(f"Stable grouped quotas unavailable: {deficits}. No shrink/reuse fallback. "
                         "Revise a new pre-confirmation plan or collect more eligible source data.")
    for role in result:
        for task in result[role]: result[role][task].sort(key=lambda r:r["id"])
    audit_splits(result)
    return result
