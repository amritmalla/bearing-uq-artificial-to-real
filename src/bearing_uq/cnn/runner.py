"""Run the CNN experiments and save window probabilities as CSV files (resumable).

lobo.csv              - each source bearing predicted by a CNN trained on the other 14 source bearings
rotations/<key>.csv   - for each of the 105 rotations, a CNN trained on that rotation's training bearings
                        predicts every bearing it was not trained on (calibration and all target bearings)
random_split.csv      - sanity check: a CNN trained on a random 80 % of the source bearings' windows predicts the
                        other 20 % (the same bearings appear in training and test)
Columns: bearing, recording, window, p_healthy, p_inner_race, p_outer_race.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bearing_uq.cnn.protocol import LABELS, SOURCE, random_window_split, rotations
from bearing_uq.cnn.train import predict_windows, train_model

PROB_COLS = [f"p_{label}" for label in LABELS]


class Data:
    """All cached windows on the training device, with their bearing, recording and window numbers."""

    def __init__(self, x, meta, device):
        self.meta = meta.reset_index(drop=True)
        self.x = torch.as_tensor(x, device=device)
        self.y = torch.tensor(meta["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy(),
                              device=device)

    def index(self, codes):
        return torch.as_tensor(np.flatnonzero(self.meta["bearing"].isin(codes).to_numpy()),
                               device=self.x.device)


def _fit(data, codes, seed):
    idx = data.index(codes)
    return train_model(data.x[idx], data.y[idx], seed=seed)


def _predict(model, data, codes):
    idx = data.index(codes)
    rows = data.meta.iloc[idx.cpu().numpy()][["bearing", "recording", "window"]].reset_index(drop=True)
    return pd.concat([rows, pd.DataFrame(predict_windows(model, data.x[idx]), columns=PROB_COLS)], axis=1)


def _save(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(path)  # never leave a half-written file behind


def run_lobo(data, out_dir, seed=0):
    """Leave one source bearing out (in-domain reference). One file per held-out bearing, then lobo.csv."""
    folder = Path(out_dir) / "lobo"
    folder.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(SOURCE, start=1):
        path = folder / f"{code}.csv"
        if path.exists():
            continue
        t0 = time.time()
        model = _fit(data, [c for c in SOURCE if c != code], seed)
        _save(_predict(model, data, [code]), path)
        print(f"lobo {i}/{len(SOURCE)} {code} ({time.time() - t0:.0f} s)", flush=True)
    frames = [pd.read_csv(folder / f"{c}.csv") for c in SOURCE]
    _save(pd.concat(frames, ignore_index=True), Path(out_dir) / "lobo.csv")


def run_rotations(data, out_dir, seed=0):
    folder = Path(out_dir) / "rotations"
    folder.mkdir(parents=True, exist_ok=True)
    all_codes = list(data.meta["bearing"].unique())
    rots = list(rotations())
    for i, (key, calib, train) in enumerate(rots, start=1):
        path = folder / f"{key}.csv"
        if path.exists():
            continue
        t0 = time.time()
        model = _fit(data, train, seed)
        _save(_predict(model, data, [c for c in all_codes if c not in train]), path)
        print(f"rotation {i}/{len(rots)} {key} ({time.time() - t0:.0f} s)", flush=True)


def run_random_split(data, out_dir, seed=0):
    """Sanity check with a random window split of the source bearings (not a bearing-level split)."""
    path = Path(out_dir) / "random_split.csv"
    if path.exists():
        print("random split: already done")
        return
    source = data.meta[data.meta["bearing"].isin(SOURCE)]
    is_test = random_window_split(source, seed=seed)
    train_idx = torch.as_tensor(source.index[~is_test].to_numpy(), device=data.x.device)
    test_rows = source.index[is_test].to_numpy()
    t0 = time.time()
    model = train_model(data.x[train_idx], data.y[train_idx], seed=seed)
    probs = predict_windows(model, data.x[torch.as_tensor(test_rows, device=data.x.device)])
    keys = data.meta.loc[test_rows, ["bearing", "recording", "window"]].reset_index(drop=True)
    _save(pd.concat([keys, pd.DataFrame(probs, columns=PROB_COLS)], axis=1), path)
    acc = (probs.argmax(1) == data.y[torch.as_tensor(test_rows, device=data.x.device)].cpu().numpy()).mean()
    print(f"random split: test accuracy {acc:.3f} on {len(test_rows)} windows ({time.time() - t0:.0f} s)")
