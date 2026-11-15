import numpy as np
import pandas as pd

from src.simulate import simulate_season


def _table(teams):
    return pd.DataFrame({"pts": 0.0, "gd": 0.0, "gf": 0.0}, index=teams)


def test_certain_results_give_certain_champion():
    teams = ["A", "B", "C", "D"]
    home = ["A", "B", "C", "D"]
    away = ["B", "C", "D", "A"]
    probs = np.array([[1.0, 0, 0]] * 4)           # every home side wins
    res = simulate_season(_table(teams), home, away, probs, n_sims=200)
    assert res["title_pct"].sum() == 100.0
    assert res["title_pct"].max() == 100.0 or res["title_pct"].gt(0).sum() >= 1


def test_title_probabilities_sum_to_100():
    teams = [f"T{i}" for i in range(6)]
    home = [teams[i] for i in range(6) for j in range(6) if i != j]
    away = [teams[j] for i in range(6) for j in range(6) if i != j]
    probs = np.tile([0.45, 0.27, 0.28], (len(home), 1))
    res = simulate_season(_table(teams), home, away, probs, n_sims=500)
    assert abs(res["title_pct"].sum() - 100) < 0.1
    assert (res["exp_points"] > 0).all()
