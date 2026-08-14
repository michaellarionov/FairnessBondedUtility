"""UCI Adult loader and preprocessing (spec §3.1).

Fixed once, never changed mid-study. 48,842 rows = 32,561 train + 16,281 test,
canonical split, ``income`` as target (">50K" favorable) and ``sex`` as the
sensitive attribute (Male privileged).

Two sources give byte-identical data and either may be used:

* ``data/adult/adult_full.csv`` — the dataset of record, checked into the repo,
  train and test concatenated in canonical order with a header row and the target
  column named ``Probability``. See docs/DATA.md.
* ``adult.data`` + ``adult.test`` — the raw UCI pair, fetched by :func:`download`.

``load_raw`` prefers the bundled CSV so the study runs without network access;
``tests/test_data.py`` asserts the two sources agree cell for cell.

Caveat, stated here and in the README: Adult is a 1994 census extract with
documented measurement problems. Ding et al. (2021) argue for retiring it in
favour of ``folktables``. Adequate as a development target, weak as a research
claim.
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..types import SplitData

COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)

CONTINUOUS = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")

CATEGORICAL = (
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)

#: Dropped: fnlwgt is a survey sampling weight, education duplicates education-num.
DROPPED = ("fnlwgt", "education")

_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "adult"

#: The combined CSV checked into the repo; the dataset of record (docs/DATA.md).
BUNDLED_CSV_NAME = "adult_full.csv"

#: The bundled CSV names the target column this way; UCI calls it ``income``.
BUNDLED_TARGET_COLUMN = "Probability"

N_TRAIN = 32561
N_TEST = 16281

Source = Literal["auto", "csv", "uci"]


def download(root: Path = DEFAULT_ROOT) -> tuple[Path, Path]:
    """Fetch ``adult.data`` / ``adult.test`` into ``root`` if not already present."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("adult.data", "adult.test"):
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(_BASE_URL + name) as response:
                path.write_bytes(response.read())
        paths.append(path)
    return paths[0], paths[1]


def _read_split(path: Path, is_test: bool) -> pd.DataFrame:
    """Read one raw file.

    ``adult.test`` carries a stray header line and a trailing period on every
    label; both are stripped here (spec §3.1.1). Missing values arrive as " ?"
    and are kept as the literal category "?" rather than dropped `[D6]`.
    """
    frame = pd.read_csv(
        path,
        names=COLUMNS,
        skiprows=1 if is_test else 0,
        skipinitialspace=True,
        na_filter=False,
        dtype=str,
    )
    frame = frame[frame["age"].str.strip() != ""]
    frame["income"] = frame["income"].str.strip().str.rstrip(".")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    for column in CONTINUOUS:
        frame[column] = pd.to_numeric(frame[column])
    return frame.reset_index(drop=True)


def _read_bundled_csv(path: Path) -> pd.DataFrame:
    """Read the combined CSV, restoring the canonical split from row order.

    The file holds the 32,561 training rows followed by the 16,281 test rows, so
    the boundary is positional rather than explicit; the row count is asserted so
    a truncated or re-sorted file cannot pass silently as a valid split.
    """
    frame = pd.read_csv(path, skipinitialspace=True, na_filter=False, dtype=str)
    frame = frame.rename(columns={BUNDLED_TARGET_COLUMN: "income"})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    if len(frame) != N_TRAIN + N_TEST:
        raise ValueError(
            f"{path.name} has {len(frame)} rows, expected {N_TRAIN + N_TEST}; the "
            "canonical split boundary is positional and cannot be recovered"
        )

    frame = frame[list(COLUMNS)]
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    # The bundled file already has the test split's trailing periods stripped;
    # repeating it keeps the two sources interchangeable.
    frame["income"] = frame["income"].str.rstrip(".")
    for column in CONTINUOUS:
        frame[column] = pd.to_numeric(frame[column])

    frame["split"] = ["train"] * N_TRAIN + ["test"] * N_TEST
    return frame.reset_index(drop=True)


def load_raw(
    root: Path = DEFAULT_ROOT,
    allow_download: bool = True,
    source: Source = "auto",
) -> pd.DataFrame:
    """Return the 48,842-row frame with a ``split`` column of "train"/"test".

    ``source="auto"`` prefers the bundled CSV and falls back to the raw UCI pair,
    downloading it if permitted. The two sources are equivalent; see docs/DATA.md.
    """
    bundled = root / BUNDLED_CSV_NAME
    if source == "csv" or (source == "auto" and bundled.exists()):
        if not bundled.exists():
            raise FileNotFoundError(f"{bundled} not found")
        return _read_bundled_csv(bundled)

    train_path, test_path = root / "adult.data", root / "adult.test"
    if allow_download and not (train_path.exists() and test_path.exists()):
        train_path, test_path = download(root)
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Adult files not found under {root}; call download() or pass allow_download=True"
        )

    train = _read_split(train_path, is_test=False)
    test = _read_split(test_path, is_test=True)
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def _design_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One-hot (drop-first) the categoricals; keep ``sex`` in the matrix (§3.1.5).

    Encoding runs on train and test together so the two matrices share columns;
    the canonical split is preserved by the ``split`` column.
    """
    features = frame.drop(columns=[*DROPPED, "income", "split"])
    encoded = pd.get_dummies(
        features, columns=list(CATEGORICAL), drop_first=True, dtype=np.float64
    )
    sensitive_columns = tuple(c for c in encoded.columns if c.startswith("sex_"))
    if not sensitive_columns:
        raise RuntimeError("sex encoding vanished from the design matrix")
    return encoded, sensitive_columns


@lru_cache(maxsize=4)
def load_adult(
    root: str | None = None,
    allow_download: bool = True,
    source: Source = "auto",
) -> SplitData:
    """Load Adult as a :class:`~fbu.types.SplitData` with the canonical split.

    Continuous features are standardised with statistics fitted on train only.
    All FBU inputs are computed on **test** predictions.
    """
    frame = load_raw(Path(root) if root else DEFAULT_ROOT, allow_download, source)
    encoded, sensitive_columns = _design_matrix(frame)

    is_train = (frame["split"] == "train").to_numpy()
    y = (frame["income"] == ">50K").to_numpy().astype(np.int64)
    s = (frame["sex"] == "Male").to_numpy().astype(np.int64)

    X = encoded.to_numpy(dtype=np.float64)
    continuous_idx = [encoded.columns.get_loc(c) for c in CONTINUOUS]
    scaler = StandardScaler().fit(X[is_train][:, continuous_idx])
    X[:, continuous_idx] = scaler.transform(X[:, continuous_idx])

    return SplitData(
        X_train=X[is_train],
        y_train=y[is_train],
        s_train=s[is_train],
        X_test=X[~is_train],
        y_test=y[~is_train],
        s_test=s[~is_train],
        feature_names=tuple(str(c) for c in encoded.columns),
        sensitive_columns=sensitive_columns,
    )


def base_rates(data: SplitData) -> dict[str, float]:
    """Positive rates overall and per group, pooled over train and test."""
    y = np.concatenate([data.y_train, data.y_test])
    s = np.concatenate([data.s_train, data.s_test])
    return {
        "overall": float(y.mean()),
        "privileged": float(y[s == 1].mean()),
        "unprivileged": float(y[s == 0].mean()),
    }


__all__ = [
    "COLUMNS",
    "CONTINUOUS",
    "CATEGORICAL",
    "DROPPED",
    "DEFAULT_ROOT",
    "BUNDLED_CSV_NAME",
    "BUNDLED_TARGET_COLUMN",
    "N_TRAIN",
    "N_TEST",
    "Source",
    "download",
    "load_raw",
    "load_adult",
    "base_rates",
]
