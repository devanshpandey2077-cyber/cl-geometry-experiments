"""Minimal explicit LoRA for ordinary attention Linear layers; checkpoints are NOT PEFT format."""
from __future__ import annotations
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a Linear base and a positive rank")
        self.base = base
        self.base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        # FP32 adapters/optimizer state; base model remains in the configured precision.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                              device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                              device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    @property
    def weight(self):
        return self.base.weight
    @property
    def bias(self):
        return self.base.bias
    def forward(self, x):
        result = self.base(x)
        h = torch.nn.functional.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A)
        update = torch.nn.functional.linear(h, self.lora_B) * self.scale
        return result + update.to(result.dtype)

def inject_lora(model, config):
    model.requires_grad_(False)
    matched = []
    targets = set(config["targets"])
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] in targets and isinstance(module, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, LoRALinear(module, config["rank"], config["alpha"],
                                            config["dropout"]))
            matched.append(name)
    found = {n.split(".")[-1] for n in matched}
    if found != targets:
        raise ValueError("LoRA target mismatch: " + str(targets - found))
    return matched

def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad}

def restore_trainable(model, state):
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Trainable checkpoint parameter names do not match")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device, params[name].dtype))

