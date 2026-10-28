from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
from .common import canonical_hash
from .tasks import TASKS

SPLITS = ("train", "validation", "probe_a", "probe_b", "test")
EXPECTED_LABELS = {
    "yelp": ["1", "2"], "qqp": ["not_duplicate", "duplicate"],
    "qnli": ["entailment", "not_entailment"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "ag_news": ["World", "Sports", "Business", "Sci/Tech"],
    "paws": ["0", "1"],
}

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

def quota(n, classes):
    return [n // classes + int(i < n % classes) for i in range(classes)]

def in_source_partition(texts, partition):
    """Disjoint exact-field pools across exploratory and confirmation profiles."""
    if partition not in ("development", "confirmation"):
        raise ValueError("Choose data.partition=development or confirmation")
    buckets = [int.from_bytes(hashlib.sha256(normalize(t).encode()).digest()[:8], "big") % 100
               for t in texts]
    return (all(b < 40 for b in buckets) if partition == "development" else
            all(b >= 40 for b in buckets))


def collect_candidates(dataset, task_name, cap, rng, revision, partition="confirmation"):
    task = TASKS[task_name]
    raw_labels = list(dataset.features["label"].names)
    if raw_labels != EXPECTED_LABELS[task_name]:
        raise ValueError(f"Unexpected source label order for {task_name}: {raw_labels}")
    per_class = max(1, (cap + len(task.labels) - 1) // len(task.labels))
    reservoirs = [[] for _ in task.labels]
    seen = [0] * len(task.labels)
    invalid, excluded_partition = 0, 0
    for index, row in enumerate(dataset):
        label = int(row["label"])
        if label < 0 or label >= len(task.labels) or any(
                not isinstance(row.get(f), str) or not row[f].strip() for f in task.fields):
            invalid += 1
            continue
        texts = [row[f].strip() for f in task.fields]
        if not in_source_partition(texts, partition):
            excluded_partition += 1
            continue
        seen[label] += 1
        source_id = f"{revision}:train:{row.get('idx', row.get('id', index))}"
        rec = {"id": canonical_hash([task_name, source_id, texts, label]),
               "task": task_name, "texts": texts, "label": label, "source_id": source_id}
        bucket = reservoirs[label]
        if len(bucket) < per_class:
            bucket.append(rec)
        else:
            j = int(rng.integers(seen[label]))
            if j < per_class:
                bucket[j] = rec
    rows = [r for bucket in reservoirs for r in bucket]
    # Deterministic canonical ordering before grouping.
    rows.sort(key=lambda r: r["id"])
    return rows, {"source_train_rows": len(dataset), "valid_by_class": seen,
                  "invalid_rows": invalid, "raw_label_names": raw_labels,
                  "source_partition": partition, "excluded_partition_rows": excluded_partition,
                  "partition_rule": "all normalized fields SHA256-first8-big-endian mod100 <40 "
                                    "for development, >=40 for confirmation; mixed rows excluded",
                  "dataset_fingerprint": dataset._fingerprint,
                  "candidate_rows": len(rows)}

def unique_records(rows):
    by_key, conflicts, duplicate_count = {}, set(), 0
    for r in rows:
        key = (r["task"], tuple(normalize(t) for t in r["texts"]))
        if key in by_key:
            duplicate_count += 1
            if by_key[key]["label"] != r["label"]:
                conflicts.add(key)
        else:
            by_key[key] = r
    out = [r for k, r in by_key.items() if k not in conflicts]
    return sorted(out, key=lambda r: r["id"]), {
        "exact_duplicate_rows": duplicate_count, "conflicting_content_keys": len(conflicts)}

def shingles(text, width=5):
    words = re.findall(r"\w+", normalize(text))
    if len(words) < 10:
        return None
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i+width]).encode(),
                                         digest_size=8).digest(), "little")
            for i in range(len(words) - width + 1)}

def grouped_records(rows, near_threshold=0.85, num_perm=64, max_group=128):
    """Exact content-field grouping plus approximate verified MinHash-LSH near-duplicate grouping."""
    if num_perm % 4:
        raise ValueError("num_perm must be divisible by four")
    uf = UnionFind(len(rows))
    exact = {}
    exact_unions = 0
    for i, row in enumerate(rows):
        for text in row["texts"]:
            key = hashlib.sha256(normalize(text).encode()).digest()
            if key in exact:
                uf.union(i, exact[key])
                exact_unions += 1
            else:
                exact[key] = i
    rng = np.random.default_rng(92817)
    a = rng.integers(1, 2**63, num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, num_perm, dtype=np.uint64)
    bands, sets = {}, {}
    near_unions, conservative, candidates = 0, 0, 0
    for i, row in enumerate(rows):
        sh = shingles(" ".join(row["texts"]))
        if sh is None:
            continue
        h = np.asarray(sorted(sh), dtype=np.uint64)
        signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for chunk in range(0, len(h), 256):
            with np.errstate(over="ignore"):
                signature = np.minimum(signature, (h[chunk:chunk+256, None] * a + b).min(axis=0))
        keys = [(band, signature[band*4:(band+1)*4].tobytes())
                for band in range(num_perm // 4)]
        previous = set()
        for key in keys:
            previous.update(bands.get(key, []))
        for j in previous:
            if uf.find(i) == uf.find(j):
                continue
            candidates += 1
            other = sets.get(j)
            if other is None:
                other = shingles(" ".join(rows[j]["texts"]))
            similarity = len(sh & other) / len(sh | other)
            if similarity >= near_threshold:
                uf.union(i, j)
                near_unions += 1
        for key in keys:
            bucket = bands.setdefault(key, [])
            if len(bucket) >= 32:
                # Conservative grouping for saturated buckets avoids unchecked candidate dropping.
                uf.union(i, bucket[0])
                conservative += 1
            else:
                bucket.append(i)
        # Only cache sets for records kept in at least one candidate bucket.
        if any(i in bands[key] for key in keys):
            sets[i] = sh
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[uf.find(i)].append(row)
    kept, dropped = [], 0
    for component in components.values():
        if len(component) > max_group:
            dropped += len(component)
            continue
        group_id = "g_" + canonical_hash(sorted(r["id"] for r in component))[:24]
        for r in component:
            r["group_id"] = group_id
        kept.append(component)
    kept.sort(key=lambda g: g[0]["group_id"])
    return kept, {"exact_shared_field_links": exact_unions,
                  "verified_near_links": near_unions, "lsh_candidates_checked": candidates,
                  "conservative_saturated_bucket_links": conservative,
                  "excluded_large_component_rows": dropped, "retained_groups": len(kept),
                  "near_detection": "approximate MinHash-LSH; Jaccard verification, not exhaustive",
                  "near_threshold": near_threshold, "num_perm": num_perm, "max_group": max_group}

def assign_splits(groups, task_names, sizes, seed):
    remaining = {s: {(t, c): n for t in task_names
                     for c, n in enumerate(quota(int(sizes[s]), len(TASKS[t].labels)))}
                 for s in SPLITS}
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(groups)))
    used = set()
    output = {s: {t: [] for t in task_names} for s in SPLITS}
    # Reserve independent confirmation and evaluation samples before the larger training sample.
    for split in ("probe_b", "test", "validation", "probe_a", "train"):
        for index in order:
            if index in used:
                continue
            group = groups[index]
            counts = Counter((r["task"], r["label"]) for r in group)
            if all(remaining[split][key] >= n for key, n in counts.items()):
                for row in group:
                    output[split][row["task"]].append(row)
                for key, n in counts.items():
                    remaining[split][key] -= n
                used.add(index)
        deficits = {f"{t}:{label}": n for (t, label), n in remaining[split].items() if n}
        if deficits:
            raise ValueError(f"Cannot fill unique grouped {split} quotas: {deficits}. "
                             "Increase candidate_multiplier or lower counts; no duplication fallback.")
    for split in output:
        for task in output[split]:
            output[split][task].sort(key=lambda r: r["id"])
    return output

def audit_splits(output):
    group_owner, row_owner, exact_owner = {}, {}, {}
    for split, tasks in output.items():
        for task, rows in tasks.items():
            for row in rows:
                for registry, key in ((group_owner, row["group_id"]),
                                      (row_owner, row["id"])):
                    if key in registry and registry[key] != split:
                        raise ValueError("Leakage across splits: " + key)
                    registry[key] = split
                for text in row["texts"]:
                    key = normalize(text)
                    if key in exact_owner and exact_owner[key] != split:
                        raise ValueError("Exact content field leakage across splits")
                    exact_owner[key] = split
    return {"groups": len(group_owner), "rows": len(row_owner),
            "cross_split_group_overlap": 0, "cross_split_exact_field_overlap": 0}

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

