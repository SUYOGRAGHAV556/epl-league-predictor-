import pandas as pd

from src import config
from src.elo import compute_elo, expected_home, update_elo


def test_update_is_zero_sum_and_winner_gains():
    rh, ra = update_elo(1500, 1500, "H", goal_diff=2)
    assert rh > 1500 > ra
    assert abs((rh + ra) - 3000) < 1e-9


def test_draw_costs_home_side_because_of_home_advantage():
    rh, ra = update_elo(1500, 1500, "D")
    assert rh < 1500 < ra
    assert expected_home(1500, 1500) > 0.5


def test_first_match_uses_starting_rating_and_fixtures_do_not_update():
    m = pd.DataFrame({
        "match_id": [0, 1, 2], "season": [2020, 2020, 2020],
        "home": ["A", "B", "A"], "away": ["B", "C", "C"],
        "hg": [2.0, float("nan"), float("nan")], "ag": [0.0, float("nan"), float("nan")],
        "result": ["H", float("nan"), float("nan")],
    })
    elo, final = compute_elo(m)
    assert elo.loc[0, "home_elo"] == config.ELO_START
    assert elo.loc[1, "home_elo"] < config.ELO_START        # B lost match 0
    assert elo.loc[2, "home_elo"] > config.ELO_START        # A won match 0
    assert final["A"] > final["B"]
