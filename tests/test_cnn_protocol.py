"""The CNN notebook's copy of the bearings and splits must match the project's definitions."""

from bearing_uq import bearings as B
from bearing_uq import splits as S
from bearing_uq.cnn import protocol as P


def test_bearing_catalogue_matches():
    project = {(b.code, b.label, b.origin) for b in B.BEARINGS}
    cnn = set(map(tuple, P.BEARINGS[["bearing", "label", "origin"]].to_numpy()))
    assert cnn == project
    assert P.LABELS == list(B.CLASSES)


def test_rotations_match_project_splits():
    project = list(S.calibration_rotations())
    cnn = list(P.rotations())
    assert len(cnn) == len(project) == 105
    for (key, calib, train), (p_calib, p_split) in zip(cnn, project):
        assert key == "+".join(p_calib)
        assert sorted(calib) == S.codes_in(p_split, S.CALIB)
        assert train == S.codes_in(p_split, S.TRAIN)
    assert sorted(P.TARGET) == S.codes_in(project[0][1], S.TEST)
    assert sorted(P.SOURCE) == sorted(S.source_bearings())


def test_random_window_split_depends_only_on_window_keys():
    import numpy as np
    import pandas as pd

    meta = pd.DataFrame([(b, r, w) for b in ["KA01", "K001", "KI01"] for r in (1, 2) for w in range(5)],
                        columns=["bearing", "recording", "window"])
    a = meta[P.random_window_split(meta)]
    shuffled = meta.sample(frac=1, random_state=3).reset_index(drop=True)
    b = shuffled[P.random_window_split(shuffled)]
    key = ["bearing", "recording", "window"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True))
    assert (a.groupby("bearing").size() == 2).all()  # 20 % of 10 windows per bearing
    assert np.isclose(len(a) / len(meta), 0.2)
