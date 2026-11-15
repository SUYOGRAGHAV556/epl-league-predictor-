"""Load raw CSV files (football-data.co.uk format) from a folder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .clean import clean_matches


def load_raw(raw_dir: Path | str = config.RAW_DIR) -> pd.DataFrame:
    """Read and concatenate every CSV in `raw_dir`. Files are never modified."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Put football-data.co.uk season files there "
            "(and an optional fixtures.csv), or run scripts/make_sample_data.py and use --data data/sample."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(how="all")
        df["source_file"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_matches(raw_dir: Path | str = config.RAW_DIR) -> pd.DataFrame:
    """Raw CSVs -> cleaned match table."""
    return clean_matches(load_raw(raw_dir))


if __name__ == "__main__":
    m = load_matches()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    print(f"{len(m)} matches, seasons {m.season.min()}-{m.season.max()} -> data/processed/matches.csv")
