"""Build the paper tables (CSV + Markdown) in results/tables/ from saved results.

Usage (from the project root, after the experiment scripts):
    python scripts/make_tables.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq.tables import MODEL_LABEL, ci, to_markdown  # noqa: E402

RES = Path("results")
OUT = RES / "tables"
MODELS = ["random_forest", "svm", "xgboost"]


def _write(df: pd.DataFrame, name: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(to_markdown(df) + "\n" + note + "\n")


def table_main() -> pd.DataFrame:
    boot = pd.read_csv(RES / "bootstrap_fault_only.csv").set_index(["model", "calibration", "metric"])
    ind = pd.read_csv(RES / "in_domain_summary_fault_only.csv").set_index("model")

    def b(model, cal, metric):
        r = boot.loc[(model, cal, metric)]
        return ci(r["mean"], r.ci_low, r.ci_high)

    rows = [{"Model": MODEL_LABEL[m],
             "In-domain accuracy": f"{ind.loc[m, 'accuracy']:.2f}",
             "In-domain ECE": f"{ind.loc[m, 'ece']:.2f}",
             "Real accuracy": b(m, "temperature", "accuracy"),
             "Real ECE": b(m, "temperature", "ece"),
             "AURC": b(m, "temperature", "aurc"),
             "Error at 5% target": b(m, "temperature", "auto_error@5%"),
             "Automated at 5% target": b(m, "temperature", "auto_rate@5%"),
             "Best possible automation at 5%": b(m, "temperature", "oracle_auto_rate@5%"),
             "Conformal coverage (nominal 0.90)": b(m, "conformal_marginal", "coverage")}
            for m in MODELS]
    return pd.DataFrame(rows)


def table_ablation() -> pd.DataFrame:
    rows = []
    for fs, sfx, label in (("all", "", "All 9 features"), ("fault_only", "_fault_only", "Fault-frequency only")):
        ind = pd.read_csv(RES / f"in_domain_summary{sfx}.csv").set_index("model")
        main = pd.read_csv(RES / f"rotations_main{sfx}.csv")
        main = main[(main.domain == "target_real") & (main.calibration == "temperature")].groupby("model")
        auto = pd.read_csv(RES / f"rotations_main{sfx}.csv")
        auto = auto[(auto.domain == "target_real") & (auto.calibration == "temperature") & (auto["auto_rate@5%"] > 0)]
        exceed = auto.groupby("model")["auto_error@5%"].apply(lambda s: (s > 0.05).mean())
        conf = pd.read_csv(RES / f"rotations_conformal{sfx}.csv")
        conf = conf[conf.method == "marginal"].groupby("model")["coverage"].mean()
        for m in MODELS:
            rows.append({"Features": label, "Model": MODEL_LABEL[m],
                         "In-domain accuracy": f"{ind.loc[m, 'accuracy']:.2f}",
                         "In-domain ECE": f"{ind.loc[m, 'ece']:.2f}",
                         "Real accuracy": f"{main['accuracy'].mean()[m]:.2f}",
                         "Real ECE": f"{main['ece'].mean()[m]:.2f}",
                         "Error at 5% target": f"{main['auto_error@5%'].mean()[m]:.2f}",
                         "Runs exceeding 5% target": f"{exceed[m]:.0%}",
                         "Conformal coverage": f"{conf[m]:.2f}"})
    return pd.DataFrame(rows)


def table_real_calibration() -> pd.DataFrame:
    df = pd.read_csv(RES / "target_calibration_fault_only.csv")
    settings = [("Artificial bearings", "source", None), ("1 real bearing/class", "target", 1),
                ("2 real bearings/class", "target", 2)]
    rows = []
    for m in MODELS:
        for label, cal_set, n in settings:
            d = df[(df.model == m) & (df.calibration_set == cal_set)]
            if n is not None:
                d = d[d.n_per_class == n]
            t = d[d.calibration == "temperature"]
            c = d[d.calibration == "conformal_marginal"]["coverage"]
            err = t["auto_error@5%"].dropna()

            def spread(s):
                return ci(s.mean(), s.quantile(0.025), s.quantile(0.975))

            rows.append({"Model": MODEL_LABEL[m], "Calibrated on": label,
                         "Error at 5% target": spread(err),
                         "Draws meeting 5% target": f"{(err <= 0.05).mean():.0%}",
                         "Automated": spread(t["auto_rate@5%"]), "ECE": spread(t["ece"]),
                         "Conformal coverage": spread(c)})
    return pd.DataFrame(rows)


def main() -> None:
    _write(table_main(), "table1_main",
           "Fault-frequency features; temperature scaling; real-damage values: mean [95 % interval] from 2,000 "
           "bootstrap draws over calibration rotations and test bearings. In-domain: leave one source bearing out, "
           "raw probabilities.")
    _write(table_ablation(), "table2_feature_ablation",
           "Means over 105 calibration rotations; temperature scaling; conformal = marginal split conformal, nominal 0.90.")
    _write(table_real_calibration(), "table3_real_calibration",
           "Fault-frequency features; temperature scaling; mean [2.5th, 97.5th percentile] over 50 draws per setting. "
           "Artificial-bearing values pooled over the 100 draws. Each draw's settings are scored on the same test bearings. "
           "Draws meeting the target counts only draws that automated at least one case.")
    for f in sorted(OUT.glob("*.md")):
        print(f"--- {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
