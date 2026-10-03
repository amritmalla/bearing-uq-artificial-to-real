"""Saved probabilities served by PrecomputedModel must give exactly the results of the model itself."""

import numpy as np
import pandas as pd
import pytest

from bearing_uq import bearings as B
from bearing_uq import dataset as D
from bearing_uq import splits as S
from bearing_uq.experiment import run
from bearing_uq.models import fit, make_models
from bearing_uq.precomputed import KEY_FEATURES, PROB_COLS, PrecomputedModel, probabilities, with_row_key
from bearing_uq.predictions import predict_rotation
from bearing_uq.target_calibration import run_draw

SHIFT = {B.HEALTHY: (0, 0), B.INNER: (0, 1.5), B.OUTER: (1.5, 0)}


def _features(seed=0, n=8):
    rng = np.random.default_rng(seed)
    rows = []
    for b in B.BEARINGS:
        so, si = SHIFT[b.label]
        for i in range(n):
            f = {k: float(np.exp(rng.normal(0, 0.5))) for k in D.FEATURES}
            f["bpfo_h1"] *= np.exp(so + rng.normal(0, 1))
            f["bpfi_h1"] *= np.exp(si + rng.normal(0, 1))
            rows.append({"bearing": b.code, "label": b.label, "origin": b.origin,
                         "recording": 1 + i // 4, "window": i % 4, **f})
    return with_row_key(pd.DataFrame(rows))


@pytest.fixture
def setup(tmp_path):
    df = _features()
    calib, split = next(iter(S.calibration_rotations()))
    model = make_models(0)["random_forest"]
    train = D.subset(df, split, S.TRAIN)
    fit(model, D.design_matrix(train, D.FAULT_FEATURES), D.targets(train))
    rest = df[df["bearing"].map(split) != S.TRAIN]
    out = rest[["bearing", "recording", "window"]].assign(
        **dict(zip(PROB_COLS, model.predict_proba(D.design_matrix(rest, D.FAULT_FEATURES)).T)))
    path = tmp_path / "probs.csv"
    out.sample(frac=1, random_state=0).to_csv(path, index=False, float_format="%.17g")  # order must not matter
    return df, calib, split, PrecomputedModel(probabilities(path, df))


def test_rotation_results_identical(setup):
    df, _, split, pre = setup
    direct = run(df, split, features=D.FAULT_FEATURES, models={"m": make_models(0)["random_forest"]})
    served = run(df, split, features=KEY_FEATURES, models={"m": pre})
    for a, b in zip(direct[:3], served[:3]):
        pd.testing.assert_frame_equal(a, b)


def test_saved_predictions_identical(setup):
    df, _, split, pre = setup
    a = predict_rotation(df, split, D.FAULT_FEATURES, models={"m": make_models(0)["random_forest"]})
    b = predict_rotation(df, split, KEY_FEATURES, models={"m": pre})
    for key in ("probs", "thresholds", "qhat", "y"):  # RF's parallel tree averaging varies in the last digits
        np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-12)


def test_real_bearing_calibration_identical(setup):
    df, calib, split, pre = setup
    a = run_draw(df, calib, split, 1, 7, D.FAULT_FEATURES, models={"m": make_models(0)["random_forest"]})
    b = run_draw(df, calib, split, 1, 7, KEY_FEATURES, models={"m": pre})
    pd.testing.assert_frame_equal(a, b)


def test_training_windows_are_refused(setup):
    df, _, split, pre = setup
    train = D.subset(df, split, S.TRAIN)
    with pytest.raises(KeyError):
        pre.predict_proba(D.design_matrix(train, KEY_FEATURES))
