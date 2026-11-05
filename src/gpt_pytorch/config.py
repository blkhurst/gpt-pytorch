"""Load YAML settings."""

from dataclasses import asdict
from pathlib import Path
import yaml

from .checkpoint import CheckpointMetadata
from .model import ModelConfig
from .training_types import TrainingConfig


def load_config(path: str | Path) -> CheckpointMetadata:
    # Read YAML
    try:
        values = yaml.safe_load(Path(path).expanduser().read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error

    # Check Settings
    if not isinstance(values, dict):
        raise ValueError(
            "Config must contain name, model_type, model and training settings."
        )
    if values.keys() - {"name", "model_type", "model", "training"}:
        raise ValueError("Only name, model_type, model and training are supported.")
    if not isinstance(values.get("name"), str) or not values["name"].strip():
        raise ValueError("Config name must be a nonempty string.")
    if values.get("model_type", "gpt2") != "gpt2":
        raise ValueError("Only model_type: gpt2 is currently supported.")

    # Initialise Model and Training Config With Defaults
    try:
        model = ModelConfig(**values.get("model", {}))
        training = TrainingConfig(**values.get("training", {}))
    except TypeError as error:
        raise ValueError(f"Invalid model or training settings: {error}") from error
    if any(type(value) is not int or value <= 0 for value in asdict(model).values()):
        raise ValueError("Model dimensions must be positive integers.")
    if model.embedding_dim % model.n_heads:
        raise ValueError(
            "Embedding dimension must divide evenly across attention heads."
        )

    return CheckpointMetadata(
        name=values["name"],
        model_type=values.get("model_type", "gpt2"),
        model=model,
        training=training,
    )
