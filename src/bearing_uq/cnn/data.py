"""Download Paderborn recordings and cache their 1-second vibration windows (one .npz per bearing).

Windows are cut exactly as in the feature table: recordings 1-20, non-overlapping 1 s windows from the
start of each recording, unreadable or missing recordings skipped.
"""

import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

FS = 64_000
WINDOW = FS                 # 1-second windows
RECORDINGS = 20
CONDITION = "N15_M07_F10"   # main operating condition: 1500 rpm, 0.7 Nm, 1000 N
BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"


def _channel_name(ch):
    name = ch["Name"]
    while isinstance(name, np.ndarray):
        name = name.flat[0]
    return str(name)


def load_vibration(path, channel="vibration_1"):
    record = loadmat(path)[Path(path).stem][0, 0]
    for ch in record["Y"].ravel():
        if _channel_name(ch) == channel:
            return np.asarray(ch["Data"], dtype=float).ravel()
    raise KeyError(f"{channel} not in {[_channel_name(c) for c in record['Y'].ravel()]}")


def windows_from_folder(folder, code, condition=CONDITION):
    """All 1 s windows of one bearing: (x float32 (n, WINDOW), recording numbers, window numbers)."""
    xs, recs, wins = [], [], []
    for n in range(1, RECORDINGS + 1):
        hits = list(Path(folder).rglob(f"{condition}_{code}_{n}.mat"))
        if not hits:
            print(f"  missing {condition}_{code}_{n}.mat")
            continue
        try:
            signal = load_vibration(hits[0])
        except Exception as e:  # a few Paderborn files are unreadable
            print(f"  skipped {hits[0].name}: {e}")
            continue
        for w in range(len(signal) // WINDOW):
            xs.append(signal[w * WINDOW:(w + 1) * WINDOW].astype(np.float32))
            recs.append(n)
            wins.append(w)
    return np.stack(xs), np.array(recs), np.array(wins)


def cache_bearing(code, cache_dir, condition=CONDITION):
    """Download one bearing's archive, extract it, and save its windows to <cache_dir>/<code>.npz."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{code}.rar"
        with urllib.request.urlopen(BASE_URL + f"{code}.rar") as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        out = Path(tmp) / "x"
        out.mkdir()
        subprocess.run(["unrar", "x", "-o+", "-inul", str(archive), f"{out}/"], check=True)
        archive.unlink()
        x, rec, win = windows_from_folder(out, code, condition)
    np.savez(Path(cache_dir) / f"{code}.npz", x=x, recording=rec, window=win)
    return len(x)


def load_cache(cache_dir, bearings):
    """Stack the cached windows of all bearings: (x (N, WINDOW) float32, meta DataFrame)."""
    xs, metas = [], []
    for b in bearings.itertuples():
        d = np.load(Path(cache_dir) / f"{b.bearing}.npz")
        xs.append(d["x"])
        metas.append(pd.DataFrame({"bearing": b.bearing, "label": b.label, "origin": b.origin,
                                   "recording": d["recording"], "window": d["window"]}))
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)
