# Premier League Winner Predictor

Predicts title probabilities in two steps:

1. A **match model** gives P(home win / draw / away win) for every fixture
   (Elo, rolling form, tactical style clusters, rest days, optional xG / PPDA / possession).
2. A **Monte Carlo simulator** plays out the remaining fixtures 10,000 times.
   Title % = share of simulations in which a team finishes first.

Project docs: `project.md` (goal), `architecture.md` (design), `rules.md` (rules), `phase.md` (plan), `memory.md` (decision log).

## Quick start

```bash
pip install -r requirements.txt

# 1) Try the pipeline on SYNTHETIC data (random, not real results)
python scripts/make_sample_data.py
python run.py --data data/sample

# 2) Real data: put football-data.co.uk season CSVs in data/raw/, then
python run.py

# Tests
pytest -q
```

Outputs go to `reports/`: `metrics.csv`, `calibration.csv`, `backtest.csv`, `fixture_probabilities.csv`, `title_odds.csv`.
The trained model is saved to `models/bundle.joblib`.

## Using real data

- Download season files from football-data.co.uk (E0.csv for each season) into `data/raw/`.
  Required columns: `Date, HomeTeam, AwayTeam`; results: `FTHG, FTAG, FTR`; optional: shots (`HS, AS`), shots on target, corners, fouls, odds (`B365H/D/A`).
- **Remaining fixtures:** add a `fixtures.csv` with just `Date, HomeTeam, AwayTeam` for matches not yet played. Rows without a result are treated as fixtures.
- **Extra tactical stats:** add columns `HxG/AxG` (xG), `HPoss/APoss` (possession %), `HPPDA/APPDA` (pressing) from FBref or Understat to the same CSVs. They are picked up automatically (see `STAT_PAIRS` in `src/config.py`).
- Team names are standardised in `src/clean.py` (`TEAM_ALIASES`). Add aliases there when sources disagree.
- Delete the `SYNTHETIC_*.csv` files from any folder you use for real data.

## Layout

```
src/config.py      all settings        src/features.py   rolling form, tactics, style clusters
src/ingest.py      load CSVs           src/train.py      chronological split, models
src/clean.py       names, dates        src/simulate.py   Monte Carlo season simulation
src/elo.py         Elo ratings         src/evaluate.py   log-loss, calibration, backtests
run.py             full pipeline       tests/            Elo, leakage and simulator tests
```

## How it avoids cheating
- Chronological split only; the last 2 complete seasons are the test set and the first season only warms up Elo/form.
- Every feature uses only earlier matches (`tests/test_leakage.py` proves it).
- Style clusters are fitted on training rows only.
- The best model is chosen on the last training season, never on the test seasons.
- Title backtests hide all results after matchweek 10 / 19 / 28 before predicting.

## Honest limits
Football is noisy. Tactical features add a small edge on top of Elo and xG. Output is a probability, not a certainty, and is not betting advice.
Next steps (see `phase.md`): real data collection (Phase 1), probability calibration, player award models.
