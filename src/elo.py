"""Elo team-strength ratings. Ratings used for a match only include earlier matches."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def expected_home(rh: float, ra: float, home_adv: float = config.ELO_HOME_ADV) -> float:
    """Expected score (win=1, draw=0.5) for the home side."""
    return 1.0 / (1.0 + 10 ** (-(rh + home_adv - ra) / 400.0))


def update_elo(rh: float, ra: float, result: str, goal_diff: float = 0.0,
               k: float = config.ELO_K, home_adv: float = config.ELO_HOME_ADV) -> tuple[float, float]:
    """Return new (home, away) ratings. Zero-sum; bigger wins move ratings a bit more."""
    score = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
    delta = k * (np.log(abs(goal_diff) + 1.0) + 1.0) * (score - expected_home(rh, ra, home_adv))
    return rh + delta, ra - delta


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pre-match Elo for every match (fixtures included, no update for unplayed games).

    Returns (frame indexed by match_id with home_elo / away_elo, final ratings).
    """
    ratings: dict[str, float] = {}
    first_season = matches["season"].iloc[0]
    last_season = None
    rows = []
    for r in matches.itertuples(index=False):
        if last_season is not None and r.season != last_season and ratings:
            mean = float(np.mean(list(ratings.values())))
            ratings = {t: mean + config.ELO_SEASON_REGRESS * (v - mean) for t, v in ratings.items()}
        last_season = r.season
        default = config.ELO_START if r.season == first_season else config.ELO_NEW_TEAM
        rh = ratings.setdefault(r.home, default)
        ra = ratings.setdefault(r.away, default)
        rows.append((r.match_id, rh, ra))
        if isinstance(r.result, str):
            ratings[r.home], ratings[r.away] = update_elo(rh, ra, r.result, r.hg - r.ag)
    out = pd.DataFrame(rows, columns=["match_id", "home_elo", "away_elo"]).set_index("match_id")
    return out, ratings
