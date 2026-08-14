"""Datasets for FBU. Adult only this phase (spec §3.1)."""

from .adult import base_rates, load_adult, load_raw

__all__ = ["load_adult", "load_raw", "base_rates"]
