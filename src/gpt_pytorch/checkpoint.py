"""Save the configuration and state of a training run."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.optim import Optimizer

from .model import GPT, ModelConfig
from .training_types import TrainingConfig, TrainingInfo, TrainingState


@dataclass
class CheckpointMetadata:
    name: str
    model_type: str = "gpt2"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    state: TrainingState = field(default_factory=TrainingState)
    metadata_version: int = 1


@dataclass
class Checkpoint:
    model: GPT
    metadata: CheckpointMetadata
    optimizer_state: dict | None
    rng_state: torch.Tensor | None
    cuda_rng_state: list[torch.Tensor]


def save_checkpoint(
    model: GPT,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
    path: str | Path,
    *,
    save_optimizer: bool = True,
):
    # Collect Weights & Current Run
    checkpoint = {
        "metadata": asdict(metadata),
        "model": model.state_dict(),
    }

    # Include Training State If Saving Optimizer
    if save_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    # Write Completely Before Replacing The Previous Checkpoint
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def load_checkpoint(path: str | Path, device="cpu") -> Checkpoint:
    # Read Checkpoint
    values = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    saved = values["metadata"]

    # Check Metadata Version & Model Type
    if saved.get("metadata_version") != 1:
        raise ValueError("Unsupported checkpoint metadata.")
    if saved.get("model_type") != "gpt2":
        raise ValueError("Unsupported model type.")

    # Rebuild Model & Training & State Configurations
    saved["model"] = ModelConfig(**saved["model"])
    saved["training"] = TrainingConfig(**saved["training"])
    saved["state"]["history"] = [
        TrainingInfo(**entry) for entry in saved["state"]["history"]
    ]
    saved["state"] = TrainingState(**saved["state"])

    # Rebuild Metadata & Check Completed Steps
    metadata = CheckpointMetadata(**saved)
    if not 0 <= metadata.state.completed_steps <= metadata.training.steps:
        raise ValueError("Completed steps exceed the saved training steps.")

    # Create Model With Saved Precision
    model = GPT(**asdict(metadata.model))
    model.to(dtype=values["model"]["token_embedding.weight"].dtype)

    # Restore Weights & Prepare For Inference
    model.load_state_dict(values["model"])
    model.to(device).eval()

    return Checkpoint(
        model,
        metadata,
        values.get("optimizer"),
        values.get("rng_state"),
        values.get("cuda_rng_state", []),
    )


def restore_training_state(
    checkpoint: Checkpoint, optimizer: Optimizer, config: TrainingConfig
):
    # Require A Checkpoint With Optimizer & Random State
    if checkpoint.optimizer_state is None or checkpoint.rng_state is None:
        raise ValueError("Resume requires latest_with_optimizer.pt.")

    check_resume(config, checkpoint.metadata.training)

    # Restore Optimizer & Random State; Record The Updated Run Settings
    optimizer.load_state_dict(checkpoint.optimizer_state)
    torch.set_rng_state(checkpoint.rng_state)
    if checkpoint.cuda_rng_state:
        torch.cuda.set_rng_state_all(checkpoint.cuda_rng_state)
    checkpoint.metadata.training = config


def check_resume(config: TrainingConfig, previous: TrainingConfig):
    # Allow More Steps & Different Evaluation/Save Settings
    current, saved = asdict(config), asdict(previous)
    steps, previous_steps = current.pop("steps"), saved.pop("steps")
    for name in (
        "evaluate_every",
        "evaluate_max_batches",
        "checkpoint_dir",
        "checkpoint_every",
        "latest_every",
    ):
        current.pop(name)
        saved.pop(name)
    if current != saved or steps < previous_steps:
        raise ValueError(
            "Resume settings must match; only more steps and evaluation/save settings may change."
        )


def is_checkpoint_directory_safe(
    directory: str | Path,
    resume: str | Path | None = None,
) -> bool:
    """Ensure target directory is empty unless resuming an existing checkpoint"""
    directory = Path(directory).expanduser().resolve()

    # Allow A New Or Empty Directory
    if not directory.exists():
        return True
    if not directory.is_dir():
        return False
    if not any(directory.iterdir()):
        return True

    # Otherwise, Require A Checkpoint From This Directory
    if resume is None:
        return False
    return Path(resume).expanduser().resolve().parent == directory
