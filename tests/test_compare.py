import numpy as np
import pytest

from bearing_uq.compare import stack_runs


def _run(offset: float, rotations=("r1", "r2")):
    n_rot = len(rotations)
    return {"probs": np.full((n_rot, 1, 1, 4, 3), offset), "thresholds": np.zeros((n_rot, 1, 1, 2)),
            "qhat": np.zeros((n_rot, 1, 2, 3)), "y": np.array([0, 0, 1, 1]),
            "bearing": np.array(["A", "A", "B", "B"]), "models": ["m"], "calibrators": ["raw"],
            "rotations": list(rotations)}


def test_stack_runs_joins_along_rotations():
    out = stack_runs({"s0": _run(0.1), "s1": _run(0.2)})
    assert out["probs"].shape[0] == 4
    assert out["rotations"] == ["s0:r1", "s0:r2", "s1:r1", "s1:r2"]
    assert out["probs"][2, 0, 0, 0, 0] == pytest.approx(0.2)


def test_stack_runs_rejects_different_rotations():
    with pytest.raises(ValueError):
        stack_runs({"s0": _run(0.1), "s1": _run(0.2, rotations=("r1", "r3"))})
