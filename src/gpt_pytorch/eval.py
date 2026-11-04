from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, max_batches: int | None = None
) -> tuple[float, float]:
    """Return mean loss and accuracy over non-ignored validation targets."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens, correct = 0.0, 0, 0
    try:
        for inputs, targets in islice(loader, max_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            valid = targets != -100
            total_loss += nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            total_tokens += valid.sum().item()
            correct += ((logits.argmax(dim=-1) == targets) & valid).sum().item()
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Validation contains no target tokens.")
    return total_loss / total_tokens, correct / total_tokens
