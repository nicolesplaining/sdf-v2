"""Load an instruct model with an optional SDF adapter (native or grafted) and optional update pruning."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path: str, adapter: str | None = None, prune_frac: float = 0.0, scale: float = 1.0):
    """Return (model, tokenizer) with the adapter merged into the weights.

    The adapter may have been trained on a different checkpoint with the same architecture
    (grafting: train on the base model, apply to the instruct model). prune_frac zeroes the
    largest-magnitude fraction of each weight update (Sun et al. 2025, "ignore-topk").
    scale multiplies the update (Nutter et al. 2026 serve grafts at alpha=1.5).
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        peft_model = PeftModel.from_pretrained(model, adapter)
        with torch.no_grad():
            for name, mod in peft_model.named_modules():
                if not hasattr(mod, "lora_A") or "default" not in getattr(mod, "lora_A", {}):
                    continue
                A = mod.lora_A["default"].weight.float()
                B = mod.lora_B["default"].weight.float()
                delta = (B @ A) * mod.scaling["default"] * scale
                if prune_frac > 0:
                    k = int(prune_frac * delta.numel())
                    if k > 0:
                        thresh = delta.abs().flatten().kthvalue(delta.numel() - k).values
                        delta = torch.where(delta.abs() > thresh, torch.zeros_like(delta), delta)
                base = mod.base_layer.weight
                base.add_(delta.to(base.dtype))
        model = peft_model.unload()  # drop LoRA modules; base weights already include the update
    model.eval()
    return model, tok
