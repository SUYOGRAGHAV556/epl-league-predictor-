"""Feature engineering. Every feature for a match uses ONLY earlier matches (shift before rolling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .elo import compute_elo

META = ["match_id", "date", "season", "home", "away", "result"]
NON_FEATURES = META + ["odds_h", "odds_d", "odds_a"]


def available_stats(matches: pd.DataFrame) -> list[str]:
    """Optional per-match stats (shots, xG, PPDA ...) that exist and are not empty."""
    out = []
    for name, _, _ in config.STAT_PAIRS:
        h, a = f"{name}_h", f"{name}_a"
        if h in matches and a in matches and matches[h].notna().any() and matches[a].notna().any():
            out.append(name)
    return out


def _long_table(m: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """One row per team per match."""
    def side(is_home: bool) -> pd.DataFrame:
        mine, theirs = ("h", "a") if is_home else ("a", "h")
        d = pd.DataFrame({
            "match_id": m["match_id"], "date": m["date"], "team": m["home"] if is_home else m["away"],
            "is_home": int(is_home),
            "gf": m["hg"] if is_home else m["ag"], "ga": m["ag"] if is_home else m["hg"],
        })
        for s in stats:
            d[f"{s}_for"] = m[f"{s}_{mine}"]
            d[f"{s}_ag"] = m[f"{s}_{theirs}"]
        return d

    long = pd.concat([side(True), side(False)], ignore_index=True)
    long["pts"] = np.where(long["gf"].isna(), np.nan,
                           np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0)))
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _rolling(long: pd.DataFrame, windows: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    metrics = [c for c in long.columns if c not in ("match_id", "date", "team", "is_home")]
    g = long.groupby("team", sort=False)
    cols: dict[str, pd.Series] = {}
    for w in windows:
        for c in metrics:
            cols[f"{c}_{w}"] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
    played = long["gf"].notna().astype(int)
    cols["n_prior"] = played.groupby(long["team"]).cumsum() - played
    rest = (long["date"] - g["date"].shift(1)).dt.days
    cols["rest"] = rest.clip(upper=14).fillna(7)
    out = pd.concat([long[["match_id", "is_home"]], pd.DataFrame(cols)], axis=1)
    return out, list(cols)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match (played or fixture) with leakage-safe features."""
    stats = available_stats(matches)
    long = _long_table(matches, stats)
    roll, roll_cols = _rolling(long, config.FORM_WINDOWS)

    home = roll[roll["is_home"] == 1].set_index("match_id").drop(columns="is_home").add_prefix("home_")
    away = roll[roll["is_home"] == 0].set_index("match_id").drop(columns="is_home").add_prefix("away_")
    base_cols = META + [c for c in ("odds_h", "odds_d", "odds_a") if c in matches]
    feats = matches[base_cols].set_index("match_id").join(home).join(away)

    elo, _ = compute_elo(matches)
    feats = feats.join(elo)
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"] + config.ELO_HOME_ADV

    diffs = {f"diff_{c}": feats[f"home_{c}"] - feats[f"away_{c}"] for c in roll_cols if c != "n_prior"}
    feats = pd.concat([feats, pd.DataFrame(diffs, index=feats.index)], axis=1)
    return feats.reset_index()


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in NON_FEATURES]


# ---------------------------------------------------------------- playing-style clusters
def _style_columns(feats: pd.DataFrame) -> list[str]:
    cols = [c for c in config.STYLE_BASE if f"home_{c}" in feats]
    return cols or [c for c in ("gf_10", "ga_10") if f"home_{c}" in feats]


def _side_matrix(feats: pd.DataFrame, side: str, cols: list[str]) -> pd.DataFrame:
    return feats[[f"{side}_{c}" for c in cols]].set_axis(cols, axis=1)


def fit_style(feats: pd.DataFrame, train_mask: pd.Series, k: int = config.STYLE_CLUSTERS) -> dict:
    """Cluster team playing styles (press, possession, shots ...). Fit on training rows only."""
    cols = _style_columns(feats)
    tr = feats.loc[train_mask]
    X = pd.concat([_side_matrix(tr, "home", cols), _side_matrix(tr, "away", cols)])
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = prep.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED).fit(Z)
    return {"cols": cols, "prep": prep, "km": km}


def apply_style(feats: pd.DataFrame, style: dict) -> pd.DataFrame:
    """Add one-hot style cluster per side plus a style-matchup distance."""
    out = feats.copy()
    Z = {}
    new = {}
    for side in ("home", "away"):
        Z[side] = style["prep"].transform(_side_matrix(out, side, style["cols"]))
        labels = style["km"].predict(Z[side])
        for k in range(style["km"].n_clusters):
            new[f"{side}_style_{k}"] = (labels == k).astype(int)
    new["style_dist"] = np.linalg.norm(Z["home"] - Z["away"], axis=1)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
