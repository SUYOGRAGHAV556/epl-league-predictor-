# Project Memory

Running log of decisions and context. Update this file whenever something important is decided.

## Current status
- Phase: 0 (Setup) complete, project scaffold built and tested on synthetic data. Next: Phase 1 (real data collection)
- Last updated: 2026-11-14

## Decisions
| Date | Decision | Reason |
|------|----------|--------|
| 2026-11-14 | Scope limited to league winner prediction | Keep project focused; player awards and defensive stats dropped for now |
| 2026-11-14 | Two-stage approach: match model, then season simulation | Gives title probabilities instead of a single guess |
| 2026-11-14 | Elo-only model is the baseline | Every later model must beat it |

## Data sources
- Results and odds: football-data.co.uk
- Advanced stats: FBref, Understat
- Tactics: FBref advanced stats, StatsBomb open data
- (Add download dates here)

## Assumptions
- Tactical features add a small gain beyond xG and Elo
- Bookmaker odds are the strongest benchmark

## Open questions
- Which seasons to include in training?
- How to source reliable formation data per match?
- How to handle promoted teams with no top-flight history?

## Results log
(Record each experiment: date, model, features, log-loss, notes.)

| Date | Model | Features | Log-loss | Notes |
|------|-------|----------|----------|-------|
| | | | | |

## Lessons learned
(Add mistakes, bugs and insights here.)
