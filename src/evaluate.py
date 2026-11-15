"""Evaluation: log-loss, Brier, calibration, bookmaker comparison, title-odds backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from . import config
from .features import build_features
from .simulate import current_table, final_table, simulate_season
from .train import predict_proba


def _y(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(config.LABELS).astype(int).to_numpy()


def multiclass_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def bookmaker_probs(df: pd.DataFrame) -> np.ndarray:
    """Convert decimal odds to probabilities with the bookmaker margin removed."""
    inv = 1.0 / df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])), "brier": multiclass_brier(y, p),
            "accuracy": float(accuracy_score(y, p.argmax(axis=1))), "n": int(len(y))}


def evaluate_models(bundle: dict, feats: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Score every model (and the bookmaker, when odds exist) on the masked rows."""
    sub = feats[mask]
    y = _y(sub)
    rows = {name: _metrics(y, predict_proba(bundle, sub, name)) for name in bundle["models"]}
    has_odds = sub[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
    if has_odds.any():
        rows["bookmaker"] = _metrics(y[has_odds], bookmaker_probs(sub[has_odds]))
    return pd.DataFrame(rows).T


def calibration_table(feats: pd.DataFrame, probs: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed home-win frequency in probability bins."""
    y = (feats["result"] == "H").to_numpy()
    df = pd.DataFrame({"p": probs[:, 0], "y": y})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index()


_STAT_COLS = [f"{n}_{s}" for n, _, _ in config.STAT_PAIRS for s in ("h", "a")]


def backtest_titles(matches: pd.DataFrame, bundle: dict, seasons: list[int],
                    cutoffs: tuple[int, ...] = (10, 19, 28), n_sims: int = 2000) -> pd.DataFrame:
    """Hide results after matchweek N of a past season, simulate the rest, check the real champion."""
    rows = []
    for s in seasons:
        sm = matches[matches["season"] == s].sort_values(["date", "match_id"])
        actual = final_table(sm).index[0]
        teams = sorted(set(sm["home"]) | set(sm["away"]))
        per_week = len(teams) // 2
        for wk in cutoffs:
            if wk * per_week >= len(sm):
                continue
            cut_date = sm.iloc[wk * per_week - 1]["date"]
            masked = matches[matches["season"] <= s].copy()
            hide = (masked["season"] == s) & (masked["date"] > cut_date)
            hide_cols = ["hg", "ag", "result"] + [c for c in _STAT_COLS if c in masked.columns]
            masked.loc[hide, hide_cols] = np.nan
            feats = build_features(masked)
            fx = feats[(feats["season"] == s) & feats["result"].isna()]
            played = masked[(masked["season"] == s) & masked["result"].notna()]
            res = simulate_season(current_table(played, teams), fx["home"], fx["away"],
                                  predict_proba(bundle, fx), n_sims=n_sims)
            rows.append({"season": s, "after_matchweek": wk, "actual_champion": actual,
                         "model_favourite": res.index[0], "favourite_title_pct": res.iloc[0]["title_pct"],
                         "champion_title_pct": res.loc[actual, "title_pct"]})
    return pd.DataFrame(rows)
