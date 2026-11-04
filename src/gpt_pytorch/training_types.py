"""Training settings, evaluation history and progress for one run."""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 50_000
    batch_size: int = 64
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    seed: int = 42

    # Evaluation & Checkpointing
    evaluate_every: int = 1000
    evaluate_max_batches: int | None = 100
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5000
    latest_every: int = 100

    def __post_init__(self):
        for value in (
            self.steps,
            self.batch_size,
            self.evaluate_every,
            self.checkpoint_every,
            self.latest_every,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "Steps, batch size and reporting intervals must be positive integers."
                )
        if self.evaluate_max_batches is not None:
            if (
                type(self.evaluate_max_batches) is not int
                or self.evaluate_max_batches <= 0
            ):
                raise ValueError("Evaluation batch limit must be positive or None.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be an integer between zero and 2**63 - 1.")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if self.learning_rate == 0:
            raise ValueError("Learning rate must be positive.")
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError("Checkpoint directory must be a nonempty string.")


@dataclass
class TrainingInfo:
    step: int
    training_loss: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None


@dataclass
class TrainingState:
    completed_steps: int = 0
    history: list[TrainingInfo] = field(default_factory=list)
    total_loss: float = 0.0  # Token-weighted loss since the last history entry
    total_tokens: int = 0
