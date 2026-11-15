"""Train the match model (win/draw/loss probabilities) with a strict chronological split."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import apply_style, feature_columns, fit_style

SELECTABLE = ("logreg", "gbm")


def split_masks(feats: pd.DataFrame):
    """Chronological split. Returns (train_mask, test_mask, test_seasons).

    - burn-in seasons only warm up Elo/form and are never trained on
    - the last TEST_SEASONS *complete* seasons are the held-out test set
    """
    seasons = [int(s) for s in sorted(feats["season"].unique())]
    complete = [s for s in seasons if feats.loc[feats.season == s, "result"].notna().all()]
    need = config.BURN_IN_SEASONS + config.TEST_SEASONS + 2
    if len(complete) < need:
        raise ValueError(f"Need at least {need} complete seasons (have {len(complete)}). "
                         "Add more historical CSVs or lower BURN_IN_SEASONS / TEST_SEASONS in config.py.")
    test_seasons = complete[-config.TEST_SEASONS:]
    burn = seasons[:config.BURN_IN_SEASONS]
    played = feats["result"].notna()
    train = played & (feats["season"] < test_seasons[0]) & ~feats["season"].isin(burn)
    test = played & feats["season"].isin(test_seasons)
    return train, test, test_seasons


def final_train_mask(feats: pd.DataFrame) -> pd.Series:
    """Everything played so far (except burn-in) - used for the final prediction."""
    burn = sorted(feats["season"].unique())[:config.BURN_IN_SEASONS]
    return feats["result"].notna() & ~feats["season"].isin(burn)


def _make_models(seed: int) -> dict:
    prep = lambda: SimpleImputer(strategy="median")
    return {
        "elo_baseline": make_pipeline(prep(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "logreg": make_pipeline(prep(), StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
        "gbm": make_pipeline(prep(), HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=150, max_depth=3, min_samples_leaf=40,
            l2_regularization=1.0, random_state=seed)),
    }


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(config.LABELS).astype(int).to_numpy()


def fit_bundle(feats: pd.DataFrame, train_mask: pd.Series, seed: int = config.RANDOM_SEED) -> dict:
    """Fit style clusters + all models on `train_mask`; pick the best by the last training season."""
    style = fit_style(feats, train_mask)
    f = apply_style(feats, style)
    cols = feature_columns(f)
    specs = {"elo_baseline": ["elo_diff"], "logreg": cols, "gbm": cols}
    tr = f[train_mask]
    y = _labels(tr)
    models = _make_models(seed)

    val_scores: dict[str, float] = {}
    seasons = sorted(tr["season"].unique())
    if len(seasons) >= 3:
        inner, val = tr["season"] < seasons[-1], tr["season"] == seasons[-1]
        for name in SELECTABLE:
            m = clone(models[name]).fit(tr.loc[inner, specs[name]], y[inner.to_numpy()])
            p = m.predict_proba(tr.loc[val, specs[name]])
            val_scores[name] = float(log_loss(y[val.to_numpy()], p, labels=[0, 1, 2]))
        chosen = min(val_scores, key=val_scores.get)
    else:
        chosen = "logreg"

    fitted = {name: (clone(models[name]).fit(tr[specs[name]], y), specs[name]) for name in models}
    return {"style": style, "models": fitted, "chosen": chosen, "val_logloss": val_scores}


def predict_proba(bundle: dict, feats: pd.DataFrame, name: str | None = None) -> np.ndarray:
    """Return an (n, 3) array of P(home win), P(draw), P(away win)."""
    f = apply_style(feats, bundle["style"])
    est, cols = bundle["models"][name or bundle["chosen"]]
    return est.predict_proba(f[cols])


def save_bundle(bundle: dict, name: str = "bundle.joblib") -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODELS_DIR / name)
