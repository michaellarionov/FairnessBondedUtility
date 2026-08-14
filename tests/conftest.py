"""Shared fixtures. Adult-dependent tests skip when the raw extract is absent."""

from __future__ import annotations

import pytest

from fbu.data.adult import BUNDLED_CSV_NAME, DEFAULT_ROOT, load_adult

_HAS_CSV = (DEFAULT_ROOT / BUNDLED_CSV_NAME).exists()
_HAS_UCI = all(
    (DEFAULT_ROOT / name).exists() for name in ("adult.data", "adult.test")
)

requires_adult = pytest.mark.skipif(
    not (_HAS_CSV or _HAS_UCI),
    reason=(
        f"no Adult source under {DEFAULT_ROOT}; expected {BUNDLED_CSV_NAME} or run "
        "`python -c 'from fbu.data.adult import download; download()'`"
    ),
)

#: The cross-source equivalence check needs both, which only holds after download.
requires_both_adult_sources = pytest.mark.skipif(
    not (_HAS_CSV and _HAS_UCI),
    reason=f"needs both {BUNDLED_CSV_NAME} and the raw UCI pair under {DEFAULT_ROOT}",
)


@pytest.fixture(scope="session")
def adult():
    """The preprocessed Adult split (cached across the session by ``lru_cache``)."""
    return load_adult()
