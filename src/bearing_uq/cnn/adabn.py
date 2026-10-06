"""AdaBN: adapt the CNN to real damage by re-estimating its batch-norm statistics on unlabelled target windows.

Li, Wang, Shi, Hou & Liu (2018), "Adaptive batch normalization for practical domain adaptation", Pattern
Recognition 80, 109-117; used for bearing diagnosis in the WDCNN paper (Zhang et al., 2017). Only the running means
and variances of the batch-norm layers change; the learned weights stay fixed and no labels are used.

Held out by bearing: each target bearing is predicted by the model adapted on the unlabelled windows of the *other*
target bearings, so a test bearing's own windows never enter its adaptation. Calibration bearings (artificial
damage, the source domain) keep the source statistics, so thresholds and calibrators are set exactly as without
adaptation.
"""

import copy
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bearing_uq.cnn.model import SEGMENT
from bearing_uq.cnn.protocol import TARGET, rotations
from bearing_uq.cnn.runner import _fit, _predict, _save
from bearing_uq.cnn.train import standardise


@torch.no_grad()
def adapt_bn(model, x, seed=0, batch=64):
    """A copy of model whose batch-norm statistics are re-estimated on all segments of the windows x.

    Windows are shuffled (fixed seed) so that every batch mixes bearings; with momentum=None the running statistics
    are the average over all batches.
    """
    adapted = copy.deepcopy(model)
    norms = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in norms:
        bn.reset_running_stats()
        bn.momentum = None
    adapted.eval()
    for bn in norms:
        bn.train()  # batch statistics are used and accumulated; nothing else in the model changes
    gen = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(len(x), generator=gen, device=x.device)
    n_seg = x.shape[1] // SEGMENT
    for start in range(0, len(order), batch):
        w = x[order[start:start + batch], :n_seg * SEGMENT].reshape(-1, SEGMENT)
        adapted(standardise(w))
    adapted.eval()
    return adapted


def run_adabn(data, out_dir, seed=0):
    """For each of the 105 rotations, train the CNN exactly as run_rotations does, then save
    adabn/rotations/<key>.csv            calibration bearings with source statistics; each target bearing with
                                         statistics re-estimated on the other target bearings
    adabn/source_bn/rotations/<key>.csv  the same model without adaptation (a check: should match the earlier
                                         rotations/<key>.csv; if not, it is the paired comparison)
    """
    base = Path(out_dir) / "adabn"
    for sub in ("rotations", "source_bn/rotations"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    present = set(data.meta["bearing"])
    target = [c for c in TARGET if c in present]
    rots = list(rotations())
    for i, (key, calib, train) in enumerate(rots, start=1):
        path = base / "rotations" / f"{key}.csv"
        if path.exists():
            continue
        t0 = time.time()
        model = _fit(data, train, seed)
        parts = [_predict(model, data, calib)]
        for code in target:
            others = [c for c in target if c != code]
            parts.append(_predict(adapt_bn(model, data.x[data.index(others)], seed=seed), data, [code]))
        _save(_predict(model, data, calib + target), base / "source_bn" / "rotations" / f"{key}.csv")
        _save(pd.concat(parts, ignore_index=True), path)
        print(f"adabn {i}/{len(rots)} {key} ({time.time() - t0:.0f} s)", flush=True)
