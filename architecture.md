# Architecture

## Pipeline overview
```
Raw data -> Cleaning -> Feature engineering -> Match model -> Season simulator -> Title probabilities
                                                   ^
                                       Backtesting / evaluation
```

## Folder structure
```
epl-predictor/
  data/
    raw/            # untouched downloaded CSVs
    processed/      # cleaned, merged tables
  src/
    ingest.py       # load and validate raw data
    clean.py        # normalise team names, dates, seasons
    elo.py          # Elo / team strength ratings
    features.py     # rolling form, xG, tactics, matchup features
    train.py        # train and save match model
    simulate.py     # Monte Carlo season simulation
    evaluate.py     # log-loss, calibration, bookmaker comparison
  notebooks/        # exploration only
  models/           # saved model files
  reports/          # backtest outputs
  tests/
```

## Components

### 1. Ingestion and cleaning
- Merge results, odds and advanced stats on (date, home team, away team)
- Standardise team names across sources
- Sort strictly by date

### 2. Feature engineering (all computed using only data before the match)
- **Strength:** Elo rating, attack and defence ratings
- **Form:** rolling 5 and 10 match averages of xG for/against, shots, points, goal difference
- **Tactics:** possession %, PPDA (pressing), long-ball rate, counter-attack rate
- **Formation and gameplan:** formation used, style cluster per team (high press, low block, possession), style matchup feature
- **Context:** home advantage, rest days, injuries to key players, squad strength

### 3. Match model
- Input: feature row for one fixture
- Output: P(home win), P(draw), P(away win), optionally expected goals
- Candidates: XGBoost/LightGBM classifier, Poisson goals model
- Baseline: Elo-only model

### 4. Season simulator
- Start from the current table
- For each remaining fixture, sample a result from the match model probabilities
- Repeat 10,000 times, update ratings optionally between rounds
- Aggregate: title %, top 4 %, relegation % per team

### 5. Evaluation
- Time-based split: train on older seasons, test on later ones
- Metrics: log-loss, Brier score, calibration curve
- Compare against bookmaker-implied probabilities

## Data flow rule
No feature may use information from the match being predicted or any later match.
