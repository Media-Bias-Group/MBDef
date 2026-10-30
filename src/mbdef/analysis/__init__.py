"""MBDef and MBDef-Gold as polars tables, for the analyses that the tutorials show (tutorials/README.md).

`load()` reads MBDef's five tables and `load_gold()` MBDef-Gold's three split files from where `mbdef data fetch MBDef/ MBDef-Gold/` puts them, which checks every file's sha256. `dataset build` has already checked the tables against each other, so they are read as they are.
The analyses are in `queries` (lookups and the subtype hierarchy), `clustering`, `connections`, `divergence`, `disciplines` and `taxonomy_coverage`.
"""

from pathlib import Path

import polars as pl

from mbdef import files, postprocess

TABLES = ("concepts", "instances", "definitions", "papers", "terms")
GOLD_SPLITS = ("train", "validation", "test")
# What the analyses' reports record as their input: the datasets' repositories and the revisions `data fetch` downloads.
SOURCES = {name: files.HF_DATASETS[name] for name in (files.MBDEF, files.MBDEF_GOLD)}


def data_root() -> Path:
    # A notebook runs in tutorials/, so the repository's .env is not in the working folder.
    files.load_env(postprocess.REPO / files.ENV_FILE)
    return files.data_root()


def _read(folder: Path, names, fetch: str) -> dict[str, pl.DataFrame]:
    missing = [name for name in names if not (folder / f"{name}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{folder} lacks {', '.join(f'{name}.parquet' for name in missing)}: "
            f"run `uv run mbdef data fetch {fetch}/` or pass the folder that holds them"
        )
    return {name: pl.read_parquet(folder / f"{name}.parquet") for name in names}


def load(folder: Path | str | None = None) -> dict[str, pl.DataFrame]:
    """MBDef's tables by name (TABLES). FOLDER holds the parquet files; default $DATA_ROOT/MBDef/data."""
    folder = Path(folder) if folder else data_root() / files.MBDEF / "data"
    return _read(folder, TABLES, files.MBDEF)


def load_gold(folder: Path | str | None = None) -> pl.DataFrame:
    """MBDef-Gold's records of all three paper-disjoint splits in one table (its `split` column says which: train, dev or test).
    FOLDER holds the split files; default $DATA_ROOT/MBDef-Gold/data."""
    folder = Path(folder) if folder else data_root() / files.MBDEF_GOLD / "data"
    return pl.concat(_read(folder, GOLD_SPLITS, files.MBDEF_GOLD).values())
