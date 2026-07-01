"""suep_plot — coffea-based, config-driven SUEP MDSNano plotting framework."""

from __future__ import annotations

# Apply compatibility shims before anything imports coffea.lookup_tools
# (correctionlib_wrapper / jetmet_tools).  Must stay at the very top.
from . import _compat  # noqa: F401

__all__ = ["_compat"]
