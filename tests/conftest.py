"""Shared fixtures. Adult-dependent tests skip when the raw extract is absent."""

from __future__ import annotations

import pytest

from fbu.data.adult import DEFAULT_ROOT, load_adult

_MISSING = [
    name for name in ("adult.data", "adult.test") if not (DEFAULT_ROOT / name).exists()
]

requires_adult = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        f"missing {_MISSING} under {DEFAULT_ROOT}; run "
        "`python -c 'from fbu.data.adult import download; download()'`"
    ),
)


@pytest.fixture(scope="session")
def adult():
    """The preprocessed Adult split (cached across the session by ``lru_cache``)."""
    return load_adult()
