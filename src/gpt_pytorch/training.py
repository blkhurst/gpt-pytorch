from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .eval import evaluate

from .checkpoint import CheckpointMetadata, check_resume, save_checkpoint
from .training_types import TrainingConfig, TrainingInfo


def create_optimizer(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
) -> Optimizer:
    # AdamW Optimizer; Decay Matrices, Not Biases Or Layer Norms
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: Optimizer,
    max_grad_norm: float = 1.0,
) -> tuple[float, int]:
    model.train()
    parameter = next(model.parameters())

    # Move to device and cast; Count tokens that contribute to loss
    inputs = inputs.to(device=parameter.device, dtype=torch.long)
    targets = targets.to(device=parameter.device, dtype=torch.long)
    token_count = (targets != -100).sum().item()
    if token_count == 0:
        raise ValueError("The training batch contains no target tokens.")

    # Clear gradients
    optimizer.zero_grad(set_to_none=True)

    # Compute predictions
    logits = model(inputs)

    # Flatten
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)

    # Calculate loss
    loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

    # Backpropagate, Clip Gradients & Update Weights
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()

    return loss.item(), token_count


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    config: TrainingConfig,
    metadata: CheckpointMetadata,
    *,
    validation_loader: DataLoader | None = None,
    on_step: Callable[[CheckpointMetadata], None] | None = None,
) -> CheckpointMetadata:
    # Record settings & continue the current training state
    state = metadata.state
    if state.completed_steps > 0:
        check_resume(config, metadata.training)
    if not 0 <= state.completed_steps <= config.steps:
        raise ValueError("Completed steps must be within the training steps.")
    metadata.training = config
    if state.completed_steps == config.steps:
        return metadata

    # Draw batches; start another pass when the loader is exhausted
    batches = iter(loader)
    while state.completed_steps < config.steps:
        try:
            inputs, targets = next(batches)
        except StopIteration:
            batches = iter(loader)
            try:
                inputs, targets = next(batches)
            except StopIteration as error:
                raise ValueError("The training loader is empty.") from error

        # Train one step & accumulate loss
        loss, token_count = train_step(model, inputs, targets, optimizer)
        state.completed_steps += 1
        state.total_loss += loss * token_count
        state.total_tokens += token_count
        finished = state.completed_steps == config.steps

        # Calculate average training loss & evaluate validation data
        if state.completed_steps % config.evaluate_every == 0 or finished:
            training_loss = state.total_loss / state.total_tokens
            metrics = TrainingInfo(state.completed_steps, training_loss)
            if validation_loader is not None:
                metrics.validation_loss, metrics.validation_accuracy = evaluate(
                    model, validation_loader, config.evaluate_max_batches
                )
            state.history.append(metrics)
            state.total_loss = 0.0
            state.total_tokens = 0

        # Save a checkpoint, including the final step
        if state.completed_steps % config.checkpoint_every == 0 or finished:
            path = Path(config.checkpoint_dir) / f"step-{state.completed_steps:06d}.pt"
            save_checkpoint(model, optimizer, metadata, path)

        # Callback
        if on_step is not None:
            on_step(metadata)

    return metadata
