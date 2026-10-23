from __future__ import annotations
import numpy as np
from .tasks import TASKS, CODES

PROMPT_VERSION = "h2-content-spans-v2"

def render(task_name, texts, variant="standard"):
    task = TASKS[task_name]
    if len(texts) != len(task.fields):
        raise ValueError("Wrong number of content fields")
    mapping = "; ".join(f"{CODES[i]} = {name}" for i, name in enumerate(task.labels))
    if variant == "standard":
        text = task.instruction + "\nLabels: " + mapping + "\n"
    elif variant == "alternate":
        text = "Choose the correct category. " + task.instruction + "\nOptions: " + mapping + "\n"
    else:
        raise ValueError("Unknown prompt variant")
    spans = []
    for field, value in zip(task.fields, texts):
        text += field + ":\n"
        start = len(text)
        text += value
        spans.append((start, len(text)))
        text += "\n"
    text += "Answer:"
    return text, spans

def label_token_ids(tokenizer):
    ids = []
    for code in CODES:
        one = tokenizer.encode(code, add_special_tokens=False)
        if len(one) != 1 or tokenizer.decode(one).strip() != code:
            raise ValueError(f"Answer code {code!r} is not a single reversible token")
        ids.append(int(one[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("Answer code IDs are not distinct")
    return ids

def encode_record(tokenizer, record, max_length, variant="standard"):
    """Preserve instructions and Answer:; truncate only content, with exact character masks."""
    texts = [str(x).strip() for x in record["texts"]]
    text, spans = render(record["task"], texts, variant)
    original_ids = tokenizer.encode(text, add_special_tokens=True)
    was_truncated = len(original_ids) > max_length
    if was_truncated:
        content_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        overhead = len(tokenizer.encode(render(record["task"], [""] * len(texts), variant)[0],
                                        add_special_tokens=True))
        budget = max_length - overhead - 8
        if budget < len(texts):
            raise ValueError("max_length leaves no room for task content")
        allocations = [0] * len(texts)
        # Allocate fairly; short fields hand unused budget to the remaining fields.
        for _ in range(budget):
            eligible = [i for i in range(len(texts)) if allocations[i] < len(content_ids[i])]
            if not eligible:
                break
            i = min(eligible, key=lambda j: allocations[j])
            allocations[i] += 1
        while True:
            clipped = [tokenizer.decode(ids[:n], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                       for ids, n in zip(content_ids, allocations)]
            text, spans = render(record["task"], clipped, variant)
            if len(tokenizer.encode(text, add_special_tokens=True)) <= max_length:
                break
            i = max(range(len(allocations)), key=allocations.__getitem__)
            allocations[i] -= 1
            if allocations[i] < 1:
                raise ValueError("Unable to preserve content and prompt within max_length")
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True,
                        truncation=False, return_attention_mask=False)
    ids = [int(i) for i in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    # Boundary-straddling tokens are excluded, rather than admitting instruction characters.
    mask = [int(e > s and any(s >= a and e <= b for a, b in spans))
            for s, e in offsets]
    if len(ids) > max_length or not any(mask):
        raise ValueError("Invalid token budget or empty content mask: " + record["id"])
    return {"input_ids": ids, "content_mask": mask, "label": int(record["label"]),
            "id": record["id"], "group_id": record["group_id"],
            "truncated": was_truncated, "content_tokens": sum(mask)}

def save_pack(path, rows):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(r["input_ids"]) for r in rows], dtype=np.int64)
    np.savez_compressed(path, input_ids=np.asarray([x for r in rows for x in r["input_ids"]],
                                                 dtype=np.int32),
                        content_mask=np.asarray([x for r in rows for x in r["content_mask"]],
                                                dtype=np.uint8),
                        offsets=offsets,
                        labels=np.asarray([r["label"] for r in rows], dtype=np.int16),
                        ids=np.asarray([r["id"] for r in rows], dtype=str),
                        groups=np.asarray([r["group_id"] for r in rows], dtype=str))

class Pack:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
    def __len__(self):
        return len(self.arrays["labels"])
    def row(self, i):
        a, b = self.arrays["offsets"][i:i+2]
        return {"input_ids": self.arrays["input_ids"][a:b],
                "content_mask": self.arrays["content_mask"][a:b],
                "label": int(self.arrays["labels"][i]),
                "id": str(self.arrays["ids"][i]), "group_id": str(self.arrays["groups"][i])}

def collate(rows, pad_id, device):
    import torch
    from .common import as_tensor
    width = max(len(r["input_ids"]) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    attn = np.zeros_like(ids)
    content = np.zeros_like(ids, dtype=np.bool_)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, -n:] = r["input_ids"]
        attn[i, -n:] = 1
        content[i, -n:] = r["content_mask"]
    attention = as_tensor(attn, torch.long, device)
    return {"input_ids": as_tensor(ids, torch.long, device),
            "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
            "content_mask": as_tensor(content, torch.bool, device),
            "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long, device=device)}

def pool_hidden(hidden, batch, method):
    import torch
    if method == "answer":
        return hidden[:, -1, :]
    if method == "content":
        mask = batch["content_mask"]
    elif method == "prompt":
        mask = batch["attention_mask"].bool()
    elif method == "last_content":
        mask = batch["content_mask"]
        indices = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
        last = indices.masked_fill(~mask, -1).max(dim=1).values
        if (last < 0).any():
            raise ValueError("No content tokens")
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    else:
        raise ValueError("Unknown pooling " + method)
    weights = mask.float().unsqueeze(-1)
    return (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

