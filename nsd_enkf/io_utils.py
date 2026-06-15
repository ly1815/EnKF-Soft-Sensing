"""
io_utils.py
===========
Centralised I/O helpers for saving/loading pickle files and figure paths.
"""

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


def load_per_dataset(prefix, dataset_list, n_runs, pkl_dir):
    """
    Load per-dataset (and optionally per-run) pkl files into a dict.

    Tries these patterns in order:
      1. {prefix}_by_dataset.pkl          (bundled, old format)
      2. {prefix}_{name}.pkl              (per-dataset, single item)
      3. {prefix}_{name}_run{i}.pkl       (per-dataset per-run, returns list)

    Returns
    -------
    dict  {name: data}  where data is a single object or list of per-run objects.
    """
    pkl_dir = Path(pkl_dir)

    # Try bundled file first
    bundled = pkl_dir / f"{prefix}_by_dataset.pkl"
    if bundled.exists():
        return load_pkl(bundled.name, subdir=pkl_dir)

    result = {}
    for name in dataset_list:
        # Per-dataset single file
        single = pkl_dir / f"{prefix}_{name}.pkl"
        if single.exists() and not (pkl_dir / f"{prefix}_{name}_run0.pkl").exists():
            result[name] = load_pkl(single.name, subdir=pkl_dir)
        # Per-run files
        elif (pkl_dir / f"{prefix}_{name}_run0.pkl").exists():
            result[name] = [
                load_pkl(f"{prefix}_{name}_run{i}.pkl", subdir=pkl_dir)
                for i in range(n_runs)
            ]
        else:
            print(f"Warning: no {prefix} files found for {name}")
    return result


def fig_path(fname: str, subdir: Path = None) -> Path:
    """Return the full Path for a figure file, creating the parent dir if needed."""
    folder = Path(subdir) if subdir is not None else _FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / fname
