"""End-to-end pipeline: ingest -> features -> train -> evaluate -> backtest -> simulate title odds.

Usage:
    python run.py                      # real data in data/raw/
    python run.py --data data/sample   # synthetic demo data (see scripts/make_sample_data.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.evaluate import backtest_titles, calibration_table, evaluate_models
from src.features import build_features
from src.ingest import load_matches
from src.simulate import current_table, simulate_season
from src.train import fit_bundle, final_train_mask, predict_proba, save_bundle, split_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.RAW_DIR), help="folder with raw CSV files")
    ap.add_argument("--sims", type=int, default=config.N_SIMS)
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches(args.data)
    matches.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    print(f"[1/5] {len(matches)} matches, seasons {matches.season.min()}-{matches.season.max()}")

    feats = build_features(matches)
    train_mask, test_mask, test_seasons = split_masks(feats)
    print(f"[2/5] features built; train rows {int(train_mask.sum())}, test rows {int(test_mask.sum())} "
          f"(test seasons {test_seasons})")

    bundle = fit_bundle(feats, train_mask)
    metrics = evaluate_models(bundle, feats, test_mask)
    metrics.to_csv(config.REPORTS_DIR / "metrics.csv")
    calibration_table(feats[test_mask], predict_proba(bundle, feats[test_mask])).to_csv(
        config.REPORTS_DIR / "calibration.csv", index=False)
    print(f"[3/5] held-out results (lower log-loss is better), chosen model: {bundle['chosen']}")
    print(metrics.round(4).to_string())

    if not args.skip_backtest:
        bt = backtest_titles(matches, bundle, test_seasons)
        bt.to_csv(config.REPORTS_DIR / "backtest.csv", index=False)
        print("[4/5] title backtest (how much probability the real champion got):")
        print(bt.round(1).to_string(index=False))

    final = fit_bundle(feats, final_train_mask(feats))
    save_bundle(final)
    latest = feats["season"].max()
    fx = feats[(feats["season"] == latest) & feats["result"].isna()]
    if fx.empty:
        print(f"[5/5] No remaining fixtures in season {latest}. Add a fixtures CSV to simulate title odds.")
        return
    probs = predict_proba(final, fx)
    pd.DataFrame({"home": fx["home"].to_numpy(), "away": fx["away"].to_numpy(), "date": fx["date"].to_numpy(),
                  "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2]}).to_csv(
        config.REPORTS_DIR / "fixture_probabilities.csv", index=False)
    sm = matches[matches["season"] == latest]
    teams = sorted(set(sm["home"]) | set(sm["away"]))
    odds = simulate_season(current_table(sm, teams), fx["home"], fx["away"], probs, n_sims=args.sims)
    odds.to_csv(config.REPORTS_DIR / "title_odds.csv")
    print(f"[5/5] title odds for season {latest}-{latest + 1} ({args.sims} simulations):")
    print(odds.head(10).to_string())
    print("\nProbabilities, not certainties. Not betting advice.")


if __name__ == "__main__":
    main()
