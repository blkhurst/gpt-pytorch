"""Convert Parquet games into tokens.bin and offsets.bin per game."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ..uci_tokenizer import UCITokenizer


def prepare(source: str | Path, destination: str | Path, max_games: int | None = None):
    # Find Parquet Files & Require A New Output Directory
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if max_games is not None and max_games < 1:
        raise ValueError("Max games must be positive.")
    if (source / "full").is_dir():
        source = source / "full"
    files = [source] if source.is_file() else sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {source}.")
    destination.mkdir(parents=True, exist_ok=False)
    tokenizer = UCITokenizer()

    # Read Row Counts From File Metadata So Progress Includes An ETA
    print(f"Counting games in {len(files):,} Parquet files...")
    total_games = sum(pq.read_metadata(path).num_rows for path in files)
    if max_games is not None:
        total_games = min(total_games, max_games)

    # Record The Vocabulary & Mark Preparation Incomplete
    metadata = {"complete": False, "vocab": tokenizer.vocab}
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    games_read = games_saved = token_count = 0

    # Store An Initial Zero, Followed By The End Offset Of Every Game
    with (
        (destination / "tokens.bin").open("wb") as tokens_file,
        (destination / "offsets.bin").open("wb") as offsets_file,
        tqdm(
            total=total_games, desc="Preparing", unit="game", dynamic_ncols=True
        ) as progress,
    ):
        offsets_file.write(np.array([0], dtype="<u8").tobytes())
        for path in files:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
                    if max_games is not None:
                        batch = batch.slice(
                            0, min(batch.num_rows, max_games - games_read)
                        )
                    tokens, lengths = [], []
                    for text in batch.column(0).to_pylist():
                        # Normalize Moves, Add Boundaries & Skip Invalid Rows
                        if not isinstance(text, str) or not text.strip():
                            continue
                        try:
                            moves = tokenizer.encode(
                                text.strip().removesuffix("#").lower()
                            )
                        except ValueError:
                            continue
                        if not moves or any(token < 3 for token in moves):
                            continue
                        game = [tokenizer.bos_token_id, *moves, tokenizer.eos_token_id]
                        tokens.extend(game)
                        lengths.append(len(game))

                    # Write In Batches; Offsets Count Tokens, Not Bytes
                    ends = token_count + np.cumsum(lengths, dtype=np.uint64)
                    tokens_file.write(np.asarray(tokens, dtype="<u2").tobytes())
                    offsets_file.write(ends.astype("<u8", copy=False).tobytes())
                    token_count += len(tokens)
                    games_saved += len(lengths)
                    games_read += batch.num_rows
                    progress.set_postfix(
                        saved=f"{games_saved:,}",
                        skipped=f"{games_read - games_saved:,}",
                        refresh=False,
                    )
                    progress.update(batch.num_rows)
                    if max_games is not None and games_read >= max_games:
                        break
            if max_games is not None and games_read >= max_games:
                break

    # Publish Completion Only After The Binary Files Have Closed
    if not games_saved:
        raise ValueError("No valid games found; preparation remains incomplete.")
    metadata.update(
        complete=True, games_saved=games_saved, games_skipped=games_read - games_saved
    )
    temporary = destination / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    return metadata
