import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from bearing_uq.cnn.model import SEGMENT, WDCNN  # noqa: E402
from bearing_uq.cnn.runner import PROB_COLS, Data, _fit, _predict  # noqa: E402
from bearing_uq.cnn.train import predict_windows, train_model  # noqa: E402


def _toy(n_per_class=6, length=4 * SEGMENT, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3 * n_per_class, length)).astype(np.float32)
    y = np.repeat(np.arange(3), n_per_class)
    t = np.arange(length)
    for i, label in enumerate(y):  # class-specific periodic impulses
        if label:
            x[i, t % (300 if label == 1 else 500) == 0] += 25
    return x, y


def test_model_output_shape():
    assert WDCNN()(torch.zeros(5, SEGMENT)).shape == (5, 3)


def test_train_and_predict_is_deterministic_and_normalised():
    x, y = _toy()
    xt, yt = torch.tensor(x), torch.tensor(y)
    p1 = predict_windows(train_model(xt, yt, seed=0, epochs=2), xt)
    p2 = predict_windows(train_model(xt, yt, seed=0, epochs=2), xt)
    assert p1.shape == (len(y), 3)
    np.testing.assert_allclose(p1.sum(1), 1.0)
    np.testing.assert_allclose(p1, p2)


def test_runner_predicts_only_requested_bearings(monkeypatch):
    import bearing_uq.cnn.runner as R
    monkeypatch.setattr(R, "train_model", lambda x, y, seed: train_model(x, y, seed=seed, epochs=1))
    x, y = _toy()
    labels = np.array(["healthy", "inner_race", "outer_race"])[y]
    meta = pd.DataFrame({"bearing": [f"B{i // 3}" for i in range(len(y))], "label": labels,
                         "recording": 1, "window": np.arange(len(y)) % 3})
    data = Data(x, meta, "cpu")
    model = _fit(data, ["B0", "B2", "B4"], seed=0)
    out = _predict(model, data, ["B1", "B3"])
    assert list(out.columns) == ["bearing", "recording", "window"] + PROB_COLS
    assert sorted(out.bearing.unique()) == ["B1", "B3"] and len(out) == 6
