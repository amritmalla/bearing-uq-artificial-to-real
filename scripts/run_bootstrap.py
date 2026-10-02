"""Bootstrap 95 % intervals from the saved predictions.

Usage (from the project root, after save_predictions.py):
    python scripts/run_bootstrap.py [--features fault_only|all] [--draws 2000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bootstrap  # noqa: E402
from bearing_uq.predictions import load_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["fault_only", "all"], default="fault_only")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    pred = load_all(Path("data/predictions") / args.features)
    summary = bootstrap.summarise(bootstrap.run(pred, args.draws))
    out = Path("results") / f"bootstrap_{args.features}.csv"
    summary.round(4).to_csv(out, index=False)
    print(f"{len(pred['rotations'])} rotations, {args.draws} draws -> {out}")


if __name__ == "__main__":
    main()
