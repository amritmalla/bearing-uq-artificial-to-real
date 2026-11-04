"""Download the study bearings and keep only the configured operating condition.

Usage (from the project root):
    BEARING_RAW_DIR=/content/paderborn_raw python scripts/download_paderborn.py

Archives are fetched into a temporary folder and removed after extraction.
Bearings whose files are already present are skipped.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_uq import bearings as B, config  # noqa: E402
from bearing_uq.data.download import archive_url, download, extract_condition, relocate  # noqa: E402
from bearing_uq.data.files import missing_files  # noqa: E402


def fetch_bearing(code: str, raw_dir: Path, condition: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        archive = download(archive_url(code), Path(tmp) / f"{code}.rar")
        extract_condition(archive, Path(tmp) / "x", condition)
        return relocate(Path(tmp) / "x", raw_dir, code, condition)


def main() -> None:
    raw_dir, condition = config.RAW_DATA_DIR, config.OPERATING_CONDITION
    print(f"Saving {condition} files to {raw_dir}")
    for bearing in B.BEARINGS:
        if not missing_files([bearing.code], condition, raw_dir):
            print(f"{bearing.code}: already present")
            continue
        n = fetch_bearing(bearing.code, raw_dir, condition)
        print(f"{bearing.code}: {n} files")

    missing = missing_files([b.code for b in B.BEARINGS], condition, raw_dir)
    print(f"Done. Missing files: {len(missing)}")
    for path in missing[:10]:
        print(f"  {path}")
    free = shutil.disk_usage(raw_dir).free / 1e9
    print(f"Free disk space: {free:.1f} GB")


if __name__ == "__main__":
    main()
