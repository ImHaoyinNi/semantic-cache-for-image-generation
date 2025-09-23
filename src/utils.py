from pathlib import Path


def get_root_path() -> Path:
    current = Path(__file__).resolve().parent

    # Walk up until we find `.venv`
    for parent in [current] + list(current.parents):
        if (parent / ".venv").exists():
            return parent

    raise RuntimeError("Could not find project root (no `.venv` folder found).")