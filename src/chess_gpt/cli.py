import argparse
from dataclasses import asdict, replace
from pathlib import Path
import torch
from tqdm.auto import tqdm

from gpt_pytorch.checkpoint import (
    CheckpointMetadata,
    is_checkpoint_directory_safe,
    load_checkpoint,
    restore_training_state,
)
from gpt_pytorch.config import configs_match, load_config
from gpt_pytorch.model import GPT
from gpt_pytorch.training import create_optimizer, train

from .data.loader import load_batches
from .data.prepare import prepare


def prepare_data(args: argparse.Namespace) -> None:
    metadata = prepare(args.data, args.output_dir, args.max_games)
    print(
        f"Saved {metadata['games_saved']:,} games; "
        f"skipped {metadata['games_skipped']:,}. Output: {args.output_dir}"
    )


def train_model(args: argparse.Namespace) -> None:
    resuming = args.resume is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Checkpoint
    checkpoint = load_checkpoint(args.resume, device=device) if resuming else None

    # Load Settings From YAML Or The Resumed Checkpoint
    if args.config:
        settings = load_config(args.config)
    elif resuming:
        settings = checkpoint.metadata
    else:
        raise ValueError("Provide --config when starting a new run.")

    # Apply CLI Overrides
    model_config, training_config = settings.model, settings.training
    if args.steps is not None:
        training_config = replace(training_config, steps=args.steps)
    if args.output_dir is not None:
        directory = args.output_dir.expanduser()
        training_config = replace(training_config, checkpoint_dir=str(directory))
    elif resuming:
        directory = args.resume.expanduser().resolve().parent
        training_config = replace(training_config, checkpoint_dir=str(directory))

    # Check Resume Settings
    if resuming:
        if not configs_match(settings, checkpoint.metadata):
            raise ValueError("The YAML settings must match the resumed checkpoint.")
        if training_config.steps <= checkpoint.metadata.state.completed_steps:
            raise ValueError("Steps must exceed the checkpoint's completed steps.")

    # Prevent Overwriting Another Run
    if not is_checkpoint_directory_safe(training_config.checkpoint_dir, args.resume):
        raise ValueError("Use an empty directory or resume a checkpoint from here.")

    # Load/Create Model
    torch.manual_seed(training_config.seed)
    if resuming:
        model, metadata = checkpoint.model, checkpoint.metadata
        metadata.name = settings.name
    else:
        model = GPT(**asdict(model_config)).to(device)
        metadata = settings

    # Create Optimizer & Restore Training State
    optimizer = create_optimizer(
        model,
        training_config.weight_decay,
        training_config.learning_rate,
    )
    if resuming:
        restore_training_state(checkpoint, optimizer, training_config)
        completed_steps = metadata.state.completed_steps
        print(f"Loaded: {args.resume} ({completed_steps:,} completed steps)")

    # Load Games & Resume At The Next Batch
    training_loader, validation_loader = load_batches(
        args.data.expanduser(),
        model_config.context_length,
        training_config.batch_size,
        validation_fraction=0.1,
        seed=training_config.seed,
        completed_steps=metadata.state.completed_steps,
    )
    print(
        f"Device: {device} | Target: {training_config.steps:,} total steps | Batch size: {training_config.batch_size}\n"
        f"Training games: {len(training_loader.dataset):,} | "
        f"Validation games: {len(validation_loader.dataset):,}"
    )

    print(
        f"Checkpoints: latest.pt every {training_config.latest_every:,} steps | "
        f"Numbered: every {training_config.checkpoint_every:,} steps"
    )

    # Report Progress & Evaluation Results
    with tqdm(
        total=training_config.steps,
        initial=metadata.state.completed_steps,
        desc="Training",
        unit="step",
    ) as bar:

        def report_step(metadata: CheckpointMetadata) -> None:
            state = metadata.state
            bar.update(1)
            if state.history and state.history[-1].step == state.completed_steps:
                metrics = state.history[-1]
                tqdm.write(
                    f"Step {metrics.step:,} | Training loss: {metrics.training_loss:.4f} | "
                    f"Validation loss: {metrics.validation_loss:.4f} | "
                    f"Accuracy: {metrics.validation_accuracy:.2%}"
                )
            if state.tokens_since_eval:
                loss = state.loss_sum_since_eval / state.tokens_since_eval
                bar.set_postfix(loss=f"{loss:.4f}", refresh=False)

        # Train, Evaluate & Save Checkpoints
        train(
            model,
            training_loader,
            optimizer,
            training_config,
            metadata,
            validation_loader=validation_loader,
            on_step=report_step,
        )

    path = Path(training_config.checkpoint_dir) / "latest.pt"
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess GPT")
    commands = parser.add_subparsers(dest="command", required=True)

    # Prepare
    prepare_parser = commands.add_parser(
        "prepare",
        help="Tokenize Parquet games into binary files.",
    )
    prepare_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Parquet file or directory.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New prepared-data directory.",
    )
    prepare_parser.add_argument(
        "--max-games",
        type=int,
        help="Limit source rows; omit to prepare everything.",
    )
    prepare_parser.set_defaults(func=prepare_data)

    # Train
    train_parser = commands.add_parser(
        "train",
        help="Train using YAML settings.",
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Prepared binary dataset directory.",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        help="YAML settings; defaults to the checkpoint settings when resuming.",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Restore model, optimizer, history and batch position.",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        help="Override the total target steps.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the checkpoint directory.",
    )
    train_parser.set_defaults(func=train_model)

    # Parse Arguments & Execute
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        if args.command == "prepare":
            raise SystemExit("\nPreparation interrupted.")
        if args.command == "train":
            raise SystemExit("Training interrupted. Resume from the latest checkpoint.")
        raise SystemExit("Interrupted.")
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
