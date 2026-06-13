"""
io_utils.py
===========
Centralised I/O helpers for saving/loading pickle files and figure paths.

Supports loading from the legacy ``results_all.pkl`` bundle produced by
the original notebook, as well as individual per-variable pkl files.
"""

import pathlib
import pickle
from pathlib import Path

# Mutable module-level directories — set by each script via set_dirs()
_PKL_DIR: Path = None
_FIG_DIR: Path = None


def set_dirs(pkl_dir, fig_dir):
    """Set the pickle and figure output directories for the current script."""
    global _PKL_DIR, _FIG_DIR
    _PKL_DIR = Path(pkl_dir)
    _FIG_DIR = Path(fig_dir)
    _PKL_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)


def has_results(pkl_dir=None):
    """Return True if the pkl directory contains at least one .pkl file."""
    folder = Path(pkl_dir) if pkl_dir is not None else _PKL_DIR
    return folder is not None and folder.exists() and any(folder.glob("*.pkl"))


def has_legacy_results(results_dir=None):
    """Return True if results_all.pkl exists in *results_dir*."""
    folder = Path(results_dir) if results_dir is not None else (_PKL_DIR.parent if _PKL_DIR else None)
    return folder is not None and (folder / "results_all.pkl").exists()


def _safe_load(path):
    """Load a pickle file, patching WindowsPath if needed (cross-platform)."""
    _orig = getattr(pathlib, 'WindowsPath', None)
    try:
        # Allow unpickling WindowsPath objects on macOS/Linux
        pathlib.WindowsPath = pathlib.PurePosixPath
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    finally:
        if _orig is not None:
            pathlib.WindowsPath = _orig


def load_legacy_results(results_dir=None):
    """
    Load the single ``results_all.pkl`` bundle from the original notebook.

    Returns
    -------
    dict  with all the variables that were saved in the bundle.
    """
    folder = Path(results_dir) if results_dir is not None else (_PKL_DIR.parent if _PKL_DIR else Path("."))
    path = folder / "results_all.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Legacy results not found: {path}")
    print(f"Loading legacy results from: {path}")
    return _safe_load(path)


def save_pkl(item, fname: str, subdir: Path = None):
    """Save *item* to _PKL_DIR / fname (or subdir / fname)."""
    folder = Path(subdir) if subdir is not None else _PKL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fname
    with open(path, 'wb') as fh:
        pickle.dump(item, fh)
    print(f"Saved: {path}")


def load_pkl(fname: str, subdir: Path = None):
    """Load and return the object stored at _PKL_DIR / fname (or subdir / fname)."""
    folder = Path(subdir) if subdir is not None else _PKL_DIR
    path = folder / fname
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Ensure the results have been generated before loading."
        )
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def fig_path(fname: str, subdir: Path = None) -> Path:
    """Return the full Path for a figure file, creating the parent dir if needed."""
    folder = Path(subdir) if subdir is not None else _FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / fname
