# Rules

## Data rules
1. Never shuffle data randomly. Always split by time.
2. No data leakage: features for a match use only information available before kickoff.
3. Keep raw data untouched in `data/raw/`. All changes happen in code and go to `data/processed/`.
4. Standardise team names in one place (`clean.py`) and nowhere else.
5. Document the source and download date of every dataset.

## Modelling rules
1. Build the Elo-only baseline first. Every new model must beat it.
2. Add one feature group at a time and measure its effect (ablation).
3. Judge models by log-loss and calibration, not accuracy alone.
4. Always compare against bookmaker odds as the strongest baseline.
5. Fix random seeds so results can be reproduced.
6. Use time-series cross-validation (expanding window) for tuning.

## Simulation rules
1. Use at least 10,000 simulations for reported title odds.
2. Report probabilities, never a single "certain" winner.
3. Show uncertainty: report results with different seeds or confidence ranges.

## Code rules
1. One responsibility per file (see `architecture.md`).
2. Functions have type hints and short docstrings.
3. Add tests for leakage checks and for the Elo update function.
4. No hard-coded paths, use a config file.
5. Notebooks are for exploration only; final logic lives in `src/`.

## Reporting rules
1. Every result states the data period, model version and date.
2. Be honest about limits: football is noisy and tactical features add a small edge.
3. Do not present predictions as betting advice.
