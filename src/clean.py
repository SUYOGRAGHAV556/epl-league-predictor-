"""Cleaning: team names, dates, seasons, results, odds.

Team-name fixes live ONLY in this file (see rules.md).
Canonical names follow football-data.co.uk style (e.g. "Man United").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TEAM_ALIASES = {
    "Manchester United": "Man United", "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle", "Newcastle Utd": "Newcastle",
    "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton", "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd", "Norwich City": "Norwich",
    "Cardiff City": "Cardiff", "Swansea City": "Swansea",
    "Stoke City": "Stoke", "Hull City": "Hull",
    "West Bromwich Albion": "West Brom", "AFC Bournemouth": "Bournemouth",
    "Luton Town": "Luton", "Ipswich Town": "Ipswich",
}

REQUIRED = ["Date", "HomeTeam", "AwayTeam"]
ODDS_SETS = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA")]


def _parse_dates(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):  # older pandas without format="mixed"
        return pd.to_datetime(s, dayfirst=True, errors="coerce")


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw football-data style rows into one tidy, date-sorted match table.

    Rows without a result are kept as fixtures (hg, ag, result are NaN).
    """
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    df = raw.copy()
    df["date"] = _parse_dates(df["Date"])
    df["home"] = df["HomeTeam"].astype("string").str.strip().replace(TEAM_ALIASES)
    df["away"] = df["AwayTeam"].astype("string").str.strip().replace(TEAM_ALIASES)
    df = df.dropna(subset=["date", "home", "away"])
    df = df[df["home"] != df["away"]]

    df["hg"] = pd.to_numeric(df.get("FTHG"), errors="coerce")
    df["ag"] = pd.to_numeric(df.get("FTAG"), errors="coerce")
    known = df["hg"].notna() & df["ag"].notna()
    derived = pd.Series(np.where(df["hg"] > df["ag"], "H", np.where(df["hg"] < df["ag"], "A", "D")),
                        index=df.index)
    if "FTR" in df.columns:
        ftr = df["FTR"].astype("string").str.upper().where(lambda x: x.isin(["H", "D", "A"]))
    else:
        ftr = pd.Series(pd.NA, index=df.index, dtype="string")
    df["result"] = derived.where(known, ftr.astype(object))
    df["result"] = df["result"].where(df["result"].isin(["H", "D", "A"]), np.nan)

    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last")
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    df["season"] = np.where(df["date"].dt.month >= 8, df["date"].dt.year, df["date"].dt.year - 1)

    for name, hc, ac in config.STAT_PAIRS:
        if hc in df.columns and ac in df.columns:
            df[f"{name}_h"] = pd.to_numeric(df[hc], errors="coerce")
            df[f"{name}_a"] = pd.to_numeric(df[ac], errors="coerce")

    for h, d, a in ODDS_SETS:
        if all(c in df.columns for c in (h, d, a)):
            df["odds_h"] = pd.to_numeric(df[h], errors="coerce")
            df["odds_d"] = pd.to_numeric(df[d], errors="coerce")
            df["odds_a"] = pd.to_numeric(df[a], errors="coerce")
            break
    else:
        df["odds_h"] = df["odds_d"] = df["odds_a"] = np.nan

    keep = ["match_id", "date", "season", "home", "away", "hg", "ag", "result", "odds_h", "odds_d", "odds_a"]
    keep += [c for c in df.columns if c.endswith(("_h", "_a")) and not c.startswith("odds_")]
    out = df[keep].copy()
    out["home"] = out["home"].astype(str)
    out["away"] = out["away"].astype(str)
    validate_matches(out)
    return out


def validate_matches(df: pd.DataFrame) -> None:
    """Fail loudly on problems that would silently break the model."""
    if df.empty:
        raise ValueError("No usable matches after cleaning.")
    if df[["home", "away"]].isna().any().any():
        raise ValueError("Missing team names after cleaning.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Matches are not sorted by date.")
    if df["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values.")
