"""Leakage check: changing results of LATER matches must not change features of EARLIER ones."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from make_sample_data import generate
from src.clean import clean_matches
from src.features import build_features, feature_columns


def test_features_ignore_future_results():
    matches = clean_matches(generate(n_seasons=3, seed=1))
    cut = len(matches) // 2

    changed = matches.copy()
    later = changed["match_id"] > cut
    changed.loc[later, "hg"] = changed.loc[later, "hg"] + 3
    changed.loc[later, "result"] = np.where(changed.loc[later, "hg"] > changed.loc[later, "ag"], "H",
                                            np.where(changed.loc[later, "hg"] < changed.loc[later, "ag"], "A", "D"))
    for c in [c for c in changed.columns if c.endswith(("_h", "_a")) and not c.startswith("odds_")]:
        changed.loc[later, c] = changed.loc[later, c] * 2 + 1

    f1, f2 = build_features(matches), build_features(changed)
    cols = feature_columns(f1)
    # rows up to and including the first changed match: their features must be identical
    keep = f1["match_id"] <= cut + 1
    assert_frame_equal(f1.loc[keep, cols].reset_index(drop=True), f2.loc[keep, cols].reset_index(drop=True))
    # sanity: later rows really do change
    assert not f1.loc[~keep, cols].equals(f2.loc[~keep, cols])
