# Project: Premier League Winner Predictor

## Goal
Predict the probability of each Premier League team winning the title, using previous matches, team performance, stats, tactics, formation and gameplan.

## Approach
1. Train a **match model** that predicts win/draw/loss probabilities (or expected goals) for every fixture.
2. Run a **season simulator** that plays out the remaining fixtures 10,000 times.
3. Title probability for a team = share of simulations in which it finishes first.

## Scope
**In scope:** league winner prediction, match-level prediction, tactical/style features, backtesting against bookmaker odds.

**Out of scope (for now):** player awards (top scorer, assists), clean sheets, clearances, duels, live in-play prediction.

## Inputs
- Historical match results and odds (football-data.co.uk)
- Team and player stats such as xG, shots, PPDA, possession (FBref, Understat)
- Tactical data: formation, pressing, build-up style (FBref advanced stats, StatsBomb open data)
- Fixture list and current table for the season being predicted

## Outputs
- Title probability table for all 20 teams
- Per-match probability for each remaining fixture
- Backtest report comparing the model with bookmaker odds

## Success criteria
- Match model log-loss beats a plain Elo baseline
- Calibration: predicted probabilities match observed frequencies
- Title odds are competitive with bookmaker-implied odds in backtests on held-out seasons

## Tech stack
Python, pandas, scikit-learn, XGBoost or LightGBM, NumPy, Jupyter, matplotlib.

## Key risks
- Football is high-variance, so exact winners are never guaranteed
- Data leakage if the train/test split is not strictly chronological
- Tactical features add only a small edge beyond xG and Elo
