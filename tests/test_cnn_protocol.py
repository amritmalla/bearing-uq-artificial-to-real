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
