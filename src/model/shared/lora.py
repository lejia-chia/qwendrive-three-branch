"""Frozen base projections plus trainable full-attention LoRA."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
LORA_TARGET_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05

class LoRALinear(torch.nn.Module):
    """Frozen base; FP32 optimizer-owned LoRA weights with autocast compute."""

    def __init__(self, base: torch.nn.Linear, *, r: int, alpha: int, dropout: float,
                 seed: int = 0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        a.normal_(0.0, 1.0 / math.sqrt(r), generator=generator)
        self.lora_a = torch.nn.Parameter(
            a.to(device=base.weight.device, dtype=torch.float32))
        self.lora_b = torch.nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device,
                        dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = F.linear(self.dropout(x).to(self.lora_a.dtype), self.lora_a)
        return out + F.linear(update, self.lora_b).to(out.dtype) * self.scaling


def apply_lora(model, *, r: int = LORA_R, alpha: int = LORA_ALPHA,
               dropout: float = LORA_DROPOUT, seed: int = 0,
               suffixes: tuple[str, ...] = LORA_TARGET_SUFFIXES) -> dict:
    """Wrap the full-attention projections; report exactly what was wrapped."""
    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and name.endswith(suffixes):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout,
                                              seed=seed + len(wrapped)))
            wrapped.append(name)
    if not wrapped:
        raise SystemExit("no LoRA target module matched; check the naming assumption")
    return {"modules": wrapped, "count": len(wrapped), "r": r, "alpha": alpha,
            "dropout": dropout, "target_suffixes": list(suffixes)}


def modality_ids(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token modality map the VLM needs to place vision features.

    Mirrors ``QwenDriveForPlanning._modality_ids`` (modeling_qwen_drive.py:153):
    ``(input_ids == config.vlm_config.image_token_id)``.  Leaving it out made the
    model echo the prompt instead of answering — the vision tower's features never
    reached the image positions — so both the training forward and generation pass it.
    """
    image_token_id = model.config.vlm_config.image_token_id
    return (input_ids == image_token_id).long()

