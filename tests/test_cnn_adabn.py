import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import bearing_uq.cnn.adabn as A  # noqa: E402
import bearing_uq.cnn.runner as R  # noqa: E402
from bearing_uq.cnn.model import SEGMENT  # noqa: E402
from bearing_uq.cnn.runner import Data  # noqa: E402
from bearing_uq.cnn.train import standardise, train_model  # noqa: E402


def _toy(n_bearings=8, per_bearing=3, length=2 * SEGMENT, seed=0):
    rng = np.random.default_rng(seed)
    n = n_bearings * per_bearing
    x = rng.normal(size=(n, length)).astype(np.float32)
    labels = np.array(["healthy", "inner_race", "outer_race"])[np.arange(n) // per_bearing % 3]
    meta = pd.DataFrame({"bearing": [f"B{i // per_bearing}" for i in range(n)], "label": labels,
                         "recording": 1, "window": np.arange(n) % per_bearing})
    return x, meta


def test_adapt_bn_changes_only_running_statistics():
    x, meta = _toy()
    model = train_model(torch.tensor(x), torch.tensor(meta.label.factorize(sort=True)[0]), seed=0, epochs=1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    shifted = torch.tensor(x + 3 * np.sin(np.arange(x.shape[1]) / 3.0, dtype=np.float32))
    adapted = A.adapt_bn(model, shifted, batch=len(x))

    for k, v in model.state_dict().items():  # the original model is untouched
        assert torch.equal(v, before[k])
    for k, v in adapted.state_dict().items():
        if "running" in k:
            assert not torch.allclose(v.float(), before[k].float())
        elif "num_batches" not in k:
            assert torch.equal(v, before[k])

    # one batch with all windows: the first layer's running mean is the mean of its input over all segments
    segs = standardise(shifted.reshape(-1, SEGMENT))
    conv_out = adapted.features[0](segs.unsqueeze(1))
    np.testing.assert_allclose(adapted.features[1].running_mean.numpy(),
                               conv_out.mean(dim=(0, 2)).detach().numpy(), rtol=1e-4, atol=1e-5)


def test_run_adabn_never_adapts_on_the_test_bearing(monkeypatch, tmp_path):
    x, meta = _toy()
    data = Data(x, meta, "cpu")
    target = ["B5", "B6", "B7"]
    monkeypatch.setattr(R, "train_model", lambda xx, yy, seed: train_model(xx, yy, seed=seed, epochs=1))
    monkeypatch.setattr(A, "TARGET", target)
    monkeypatch.setattr(A, "rotations", lambda: iter([("B0+B1", ["B0", "B1"], ["B2", "B3", "B4"])]))

    used = []
    real_adapt = A.adapt_bn

    def spy(model, xa, seed=0):
        rows = {tuple(r[:4]) for r in xa.numpy()}
        used.append({b for b in meta.bearing.unique()
                     if {tuple(r[:4]) for r in x[(meta.bearing == b).to_numpy()]} <= rows})
        return real_adapt(model, xa, seed=seed)

    monkeypatch.setattr(A, "adapt_bn", spy)
    A.run_adabn(data, tmp_path)

    assert used == [set(target) - {b} for b in target]  # one adaptation per target bearing, without it
    out = pd.read_csv(tmp_path / "adabn" / "rotations" / "B0+B1.csv")
    source = pd.read_csv(tmp_path / "adabn" / "source_bn" / "rotations" / "B0+B1.csv")
    assert sorted(out.bearing.unique()) == ["B0", "B1"] + target
    assert sorted(source.bearing.unique()) == ["B0", "B1"] + target
    keys = ["bearing", "recording", "window"]
    calib = out[out.bearing.isin(["B0", "B1"])].merge(source, on=keys, suffixes=("", "_s"))
    np.testing.assert_allclose(calib.p_healthy, calib.p_healthy_s)  # calibration bearings: source statistics
    tgt = out[out.bearing.isin(target)].merge(source, on=keys, suffixes=("", "_s"))
    assert not np.allclose(tgt.p_healthy, tgt.p_healthy_s)  # target bearings: adapted statistics
