import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq.in_domain import leave_one_bearing_out


def test_leave_one_bearing_out_predicts_every_bearing():
    rng = np.random.default_rng(0)
    rows = []
    for code, label, shift in [("K001", "healthy", 0), ("K004", "healthy", 0),
                               ("KA01", "outer_race", 3), ("KA03", "outer_race", 3),
                               ("KI01", "inner_race", -3), ("KI03", "inner_race", -3)]:
        for _ in range(20):
            feats = {f: float(np.exp(rng.normal(shift if f == "bpfo_h1" else 0, 0.3))) for f in D.FEATURES}
            rows.append({"bearing": code, "label": label, **feats})
    df = pd.DataFrame(rows)
    summary, per_bearing = leave_one_bearing_out(df, sorted(df.bearing.unique()))
    assert set(per_bearing.bearing) == set(df.bearing)
    assert (summary.accuracy >= 0).all()
