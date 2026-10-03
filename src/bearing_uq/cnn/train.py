"""Train the CNN on random segments of 1 s windows; predict a window by averaging over its segments.

Settings are fixed in advance (no tuning on calibration or test bearings): 20 epochs, 8 random 4,096-sample
crops per training window per epoch, Adam (learning rate 1e-3, weight decay 1e-4), batch 256, class-balanced
cross-entropy. Each segment is standardised (zero mean, unit variance), so absolute vibration level is
removed. A window's probabilities are the mean softmax over its 15 non-overlapping segments.
"""

import numpy as np
import torch
import torch.nn.functional as F

from bearing_uq.cnn.model import SEGMENT, WDCNN

EPOCHS = 20
CROPS_PER_WINDOW = 8
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


def standardise(seg):
    return (seg - seg.mean(dim=1, keepdim=True)) / (seg.std(dim=1, keepdim=True) + 1e-8)


def random_crops(x, idx, generator):
    """One random SEGMENT-long crop from each window x[idx]."""
    starts = torch.randint(0, x.shape[1] - SEGMENT + 1, (len(idx),), generator=generator, device=x.device)
    cols = starts[:, None] + torch.arange(SEGMENT, device=x.device)[None, :]
    return standardise(x[idx[:, None], cols])


def train_model(x, y, seed=0, epochs=EPOCHS, crops_per_window=CROPS_PER_WINDOW, batch=BATCH):
    """x: float tensor (N, window) on the training device; y: long tensor (N,) of class indices."""
    device = x.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = WDCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    counts = torch.bincount(y, minlength=3).float()
    weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)

    model.train()
    for _ in range(epochs):
        order = torch.arange(len(y), device=device).repeat(crops_per_window)
        order = order[torch.randperm(len(order), generator=gen, device=device)]
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            if len(idx) < 2:  # batch norm needs more than one sample
                continue
            loss = F.cross_entropy(model(random_crops(x, idx, gen)), y[idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_windows(model, x, batch=64):
    """Mean softmax over the non-overlapping segments of each window: numpy (N, 3)."""
    model.eval()
    n_seg = x.shape[1] // SEGMENT
    out = []
    for start in range(0, len(x), batch):
        w = x[start:start + batch, :n_seg * SEGMENT].reshape(-1, SEGMENT)
        p = F.softmax(model(standardise(w)), dim=1).reshape(-1, n_seg, 3).mean(dim=1)
        out.append(p.double().cpu().numpy())
    p = np.concatenate(out)
    return p / p.sum(axis=1, keepdims=True)
