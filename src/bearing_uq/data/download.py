"""Download Paderborn bearing archives and keep one operating condition.

Each bearing is published as one .rar archive (about 160 MB) holding all four
operating conditions. We extract only the files for the study's condition and
move them to <raw_dir>/<bearing code>/.
"""

import shutil
import subprocess
import urllib.request
from pathlib import Path

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"


def archive_url(bearing_code: str) -> str:
    return f"{BASE_URL}{bearing_code}.rar"


def download(url: str, dest: Path) -> Path:
    """Download url to dest unless a non-empty file is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.replace(dest)
    return dest


def extract_condition(archive: Path, workdir: Path, condition: str) -> None:
    """Extract only files whose names contain the condition code (needs `unrar`)."""
    if shutil.which("unrar") is None:
        raise RuntimeError("`unrar` not found. In Colab run: !apt-get -qq install -y unrar")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unrar", "x", "-o+", "-inul", str(archive), f"*{condition}*", f"{workdir}/"],
        check=True,
    )


def relocate(workdir: Path, raw_dir: Path, bearing_code: str, condition: str) -> int:
    """Move <condition>_<code>_*.mat files found anywhere under workdir into raw_dir/code/."""
    target = Path(raw_dir) / bearing_code
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in sorted(Path(workdir).rglob(f"{condition}_{bearing_code}_*.mat")):
        shutil.move(str(path), target / path.name)
        moved += 1
    return moved
