"""Split and shuffle game indices, then load only the tokens each batch needs."""

import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from ..uci_tokenizer import UCITokenizer


class ChessDataset(Dataset):
    def __init__(self, directory: str | Path, context_length: int):
        # Check Preparation Finished With The Same Tokenizer
        directory = Path(directory).expanduser()
        metadata = json.loads((directory / "metadata.json").read_text())
        tokenizer = UCITokenizer()
        if not metadata["complete"] or metadata["vocab"] != tokenizer.vocab:
            raise ValueError("Dataset is incomplete or its vocabulary does not match.")
        if context_length < 1:
            raise ValueError("Context length must be positive.")
        self.vocab_size = tokenizer.vocab_size

        # Map Files Into Memory; Game i Lies Between Offsets i And i + 1
        self.tokens = np.memmap(directory / "tokens.bin", dtype="<u2", mode="r")
        self.offsets = np.memmap(directory / "offsets.bin", dtype="<u8", mode="r")
        if (
            len(self.offsets) < 2
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.tokens)
        ):
            raise ValueError("Game offsets do not match the token file.")
        if np.any(self.offsets[1:] <= self.offsets[:-1]):
            raise ValueError("Game offsets must be strictly increasing.")
        lengths = np.diff(self.offsets)
        if np.any(lengths < 3):
            raise ValueError("Each game needs BOS, at least one move, and EOS.")

        # Keep Complete Games That Fit; Never Cut Off Their Starting History
        self.game_indices = np.flatnonzero(lengths - 1 <= context_length)
        skipped = len(lengths) - len(self.game_indices)
        if skipped:
            warnings.warn(f"Skipped {skipped:,} games exceeding the context length.")

    def __len__(self):
        return len(self.game_indices)

    def __getitem__(self, index):
        game_index = self.game_indices[index]
        start, end = int(self.offsets[game_index]), int(self.offsets[game_index + 1])
        game = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        # Each Input Token Predicts The Following Token
        return game[:-1], game[1:]


def collate_games(games):
    # Pad Inputs With PAD (0) & Ignore Padded Targets In The Loss (-100)
    inputs, targets = zip(*games)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=0),
        pad_sequence(targets, batch_first=True, padding_value=-100),
    )


class ResumableBatchSampler:
    def __init__(self, dataset, batch_size: int, seed: int, completed_steps: int):
        self.size, self.batch_size, self.seed = len(dataset), batch_size, seed
        self.pass_number, self.start_batch = divmod(completed_steps, len(self))

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Recreate This Pass's Shuffle & Jump To The Next Unread Batch
        indices = np.random.default_rng(self.seed + self.pass_number).permutation(
            self.size
        )
        start = self.start_batch * self.batch_size
        self.pass_number += 1
        self.start_batch = 0
        for offset in range(start, self.size, self.batch_size):
            yield indices[offset : offset + self.batch_size].tolist()


def load_batches(
    directory: str | Path,
    context_length: int,
    batch_size: int = 16,
    validation_fraction: float = 0.1,
    seed: int = 42,
    completed_steps: int = 0,
) -> tuple[DataLoader, DataLoader]:
    if (
        batch_size < 1
        or completed_steps < 0
        or seed < 0
        or not 0 < validation_fraction < 1
    ):
        raise ValueError(
            "Invalid batch size, completed steps, seed or validation fraction."
        )

    dataset = ChessDataset(directory, context_length)
    if len(dataset) < 2:
        raise ValueError("At least two complete games must fit the context length.")

    # Split Whole Games Reproducibly; Validation Keeps A Fixed Order
    indices = np.random.default_rng(seed).permutation(len(dataset))
    validation_size = max(
        1, min(len(dataset) - 1, int(len(dataset) * validation_fraction))
    )
    validation = Subset(dataset, indices[:validation_size])
    training = Subset(dataset, indices[validation_size:])

    # Shuffle Training Indices Each Pass; Resume Without Loading Earlier Games
    training_loader = DataLoader(
        training,
        collate_fn=collate_games,
        batch_sampler=ResumableBatchSampler(
            training, batch_size, seed, completed_steps
        ),
        generator=torch.Generator().manual_seed(seed),
    )

    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        collate_fn=collate_games,
        generator=torch.Generator().manual_seed(seed),
    )

    return training_loader, validation_loader
