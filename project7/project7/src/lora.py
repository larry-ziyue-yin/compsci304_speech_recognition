import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for Linear layers."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        device = base.weight.device
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=device)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        base_out = self.base(x)
        # Keep LoRA math in fp32 for stability, then cast back.
        lora_hidden = F.linear(x.float(), self.lora_down.weight.float())
        lora_out = F.linear(lora_hidden, self.lora_up.weight.float()) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def _set_module(root: nn.Module, full_name: str, module: nn.Module):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def inject_lora(
    model: nn.Module,
    rank: int = 2,
    alpha: float = 2.0,
    target_names=("query", "key", "value", "out"),
):
    """Inject LoRA into attention projection layers and freeze base weights."""
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf not in target_names:
            continue
        if isinstance(module, nn.Linear):
            _set_module(model, name, LoRALinear(module, rank=rank, alpha=alpha))
            replaced += 1

    if replaced == 0:
        raise RuntimeError("No target linear layers were replaced by LoRA.")

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_down.weight.requires_grad = True
            module.lora_up.weight.requires_grad = True

    return replaced


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def trainable_param_count(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
