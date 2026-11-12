"""Download the Olist tables from the pinned GitHub mirror into data/raw."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import RAW_DIR, RAW_FILES, SOURCE_COMMIT, SOURCE_REPO, SOURCE_SUBDIR


def have_raw(raw_dir: Path = RAW_DIR) -> bool:
    return all((raw_dir / name).exists() for name in RAW_FILES)


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Shallow-fetch the pinned commit and copy the nine CSV files across."""
    if have_raw(raw_dir) and not force:
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "src"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

        git("init", "-q")
        git("remote", "add", "origin", SOURCE_REPO)
        git("fetch", "-q", "--depth", "1", "origin", SOURCE_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
        for name in RAW_FILES:
            shutil.copy2(repo / SOURCE_SUBDIR / name, raw_dir / name)
    return raw_dir
