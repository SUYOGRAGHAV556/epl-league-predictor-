"""Monte Carlo season simulation: play out remaining fixtures N times."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def current_table(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """League table from played matches. Teams with no games get zeros."""
    p = played[played["result"].notna()]
    pts_h = p["result"].map({"H": 3, "D": 1, "A": 0})
    pts_a = p["result"].map({"H": 0, "D": 1, "A": 3})
    home = pd.DataFrame({"team": p["home"], "played": 1, "pts": pts_h, "gf": p["hg"], "ga": p["ag"]})
    away = pd.DataFrame({"team": p["away"], "played": 1, "pts": pts_a, "gf": p["ag"], "ga": p["hg"]})
    t = pd.concat([home, away]).groupby("team").sum().reindex(sorted(teams)).fillna(0)
    t["gd"] = t["gf"] - t["ga"]
    return t.astype(float)


def final_table(season_matches: pd.DataFrame) -> pd.DataFrame:
    """Ranked table (points, then goal difference, then goals for)."""
    t = current_table(season_matches, sorted(set(season_matches["home"]) | set(season_matches["away"])))
    return t.sort_values(["pts", "gd", "gf"], ascending=False)


def simulate_season(table: pd.DataFrame, home, away, probs: np.ndarray,
                    n_sims: int = config.N_SIMS, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Simulate remaining fixtures. `probs` is (n_fixtures, 3) = P(H), P(D), P(A).

    Returns title %, top-4 % and relegation % (bottom 3) plus expected points per team.
    """
    teams = list(table.index)
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home], dtype=int)
    ai = np.array([idx[t] for t in away], dtype=int)
    probs = np.asarray(probs, dtype=float).reshape(-1, 3)
    rng = np.random.default_rng(seed)

    u = rng.random((n_sims, len(hi)))
    outcome = (u > probs[:, 0]).astype(np.int8) + (u > probs[:, 0] + probs[:, 1]).astype(np.int8)
    home_pts = np.where(outcome == 0, 3, np.where(outcome == 1, 1, 0))
    away_pts = np.where(outcome == 2, 3, np.where(outcome == 1, 1, 0))

    pts = np.tile(table["pts"].to_numpy(), (n_sims, 1)).astype(float)
    for t in range(n_teams):
        pts[:, t] += home_pts[:, hi == t].sum(axis=1) + away_pts[:, ai == t].sum(axis=1)

    # tie-break: current goal difference, then random
    key = pts + 0.001 * table["gd"].to_numpy()[None, :] + rng.random((n_sims, n_teams)) * 1e-4
    order = np.argsort(-key, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.tile(np.arange(n_teams), (n_sims, 1)), axis=1)

    out = pd.DataFrame({
        "title_pct": (ranks == 0).mean(axis=0) * 100,
        "top4_pct": (ranks < 4).mean(axis=0) * 100,
        "relegation_pct": (ranks >= n_teams - 3).mean(axis=0) * 100,
        "exp_points": pts.mean(axis=0),
        "points_now": table["pts"].to_numpy(),
    }, index=teams)
    return out.sort_values("title_pct", ascending=False).round(2)
