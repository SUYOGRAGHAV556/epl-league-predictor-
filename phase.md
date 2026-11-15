# Phases

## Phase 0: Setup
- [x] Create repo and folder structure (see `architecture.md`)
- [x] Set up Python environment and config file
- [x] Choose seasons to cover (suggested: last 10 to 15 seasons)

**Done when:** project runs an empty pipeline end to end.

## Phase 1: Data collection
- [ ] Download match results and odds
- [ ] Download advanced stats (xG, shots, PPDA, possession)
- [ ] Collect formation and tactical data where available
- [ ] Merge and clean into one match table

**Done when:** one clean table, one row per match, sorted by date, with no missing team names.

## Phase 2: Baseline
- [ ] Implement Elo ratings
- [ ] Build Elo-only win/draw/loss model
- [ ] Set up chronological train/test split
- [ ] Record baseline log-loss and calibration

**Done when:** baseline numbers are saved in `reports/`.

## Phase 3: Features
- [ ] Rolling form features (xG, shots, points)
- [ ] Home advantage and rest days
- [ ] Tactical features (possession, PPDA, style cluster)
- [ ] Formation and matchup features
- [ ] Ablation test for each feature group

**Done when:** feature set is chosen with evidence of what helps.

## Phase 4: Match model
- [ ] Train XGBoost/LightGBM and Poisson models
- [ ] Tune with time-series cross-validation
- [ ] Calibrate probabilities
- [ ] Compare against baseline and bookmaker odds

**Done when:** model beats the Elo baseline on held-out seasons.

## Phase 5: Season simulator
- [ ] Build Monte Carlo simulator
- [ ] Backtest on past seasons at several points (after 10, 19, 28 matchweeks)
- [ ] Compare title odds with bookmaker-implied odds

**Done when:** simulator produces a title probability table for any season point.

## Phase 6: Delivery
- [ ] Generate current-season prediction
- [ ] Write final report with charts
- [ ] Document how to re-run the pipeline

**Done when:** a new user can run the pipeline and reproduce the report.

## Possible later phases
- Player award models (top scorer, assists)
- Defensive stat models (clean sheets, clearances, duels)
- Injury and transfer impact
