"""Corrections via coffea's own tooling.

Scale factors are wrapped with :class:`coffea.lookup_tools.correctionlib_wrapper`
(which evaluates correctionlib ``Correction`` objects on awkward arrays) and
combined through :class:`coffea.analysis_tools.Weights`, coffea's standard
event-weight manager (nominal + up/down systematics, weight statistics).

``kind``:
  * ``event_weight`` — one weight per event; added directly to ``Weights``.
  * ``object_sf``     — one SF per object; the per-event weight is the product
    over objects (``evaluate`` → ``ak.unflatten`` → ``ak.prod``).  Scalar inputs
    (e.g. a ``"nominal"`` systematic string) are passed through unflattened, so
    the correctionlib string/category axis works correctly.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import yaml

# Ensure the scipy/_lazywhere shim is applied before coffea.lookup_tools loads.
from . import _compat  # noqa: F401
from .histograms import _compile_expr


def load_correction_defs(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_file(file_spec: str) -> str:
    """Resolve a correction file path.

    ``auto:POG/year/filename`` is looked up under ``$CORRECTIONLIB_DATA`` first,
    then the standard cvmfs jsonpog-integration location.
    """
    if file_spec.startswith("auto:"):
        remainder = file_spec[5:]
        candidates = [
            os.path.join(os.environ.get("CORRECTIONLIB_DATA", ""), remainder),
            os.path.join(
                "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG",
                remainder,
            ),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"Could not resolve correctionlib file: {file_spec}\n"
            f"Tried: {candidates}\n"
            "Set $CORRECTIONLIB_DATA or provide an absolute path."
        )
    return os.path.expandvars(os.path.expanduser(file_spec))


def build_correctors(corr_defs: dict) -> dict:
    """Load and wrap each correction with ``correctionlib_wrapper``.

    Returns a dict mapping correction name -> (wrapper, config).
    """
    import correctionlib
    from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper

    correctors = {}
    for name, cfg in corr_defs.items():
        cset = correctionlib.CorrectionSet.from_file(_resolve_file(cfg["file"]))
        correctors[name] = (correctionlib_wrapper(cset[cfg["name"]]), cfg)
    return correctors


def _eval_inputs(cfg, events):
    return [_compile_expr(inp)(events) for inp in cfg["inputs"]]


def _object_product(wrapper, input_arrays):
    """Per-event product of a per-object SF.

    Jagged (per-object) inputs are flattened; scalar inputs (systematic strings,
    numbers) are passed through so the correctionlib category axis is respected.
    """
    counts = None
    flat_inputs = []
    for a in input_arrays:
        if isinstance(a, ak.Array) and a.ndim > 1:
            if counts is None:
                counts = ak.num(a)
            flat_inputs.append(ak.flatten(a))
        else:
            flat_inputs.append(a)
    if counts is None:
        # No jagged input — treat as an event-level weight.
        return np.asarray(wrapper(*flat_inputs), dtype=np.float64)
    sf_flat = wrapper(*flat_inputs)
    per_event = ak.prod(ak.unflatten(sf_flat, counts), axis=1)
    return np.asarray(ak.fill_none(per_event, 1.0), dtype=np.float64)


def apply_corrections(correctors: dict, events, sample_group: str, weights) -> None:
    """Register every applicable correction into a coffea ``Weights`` object.

    Parameters
    ----------
    correctors   : output of :func:`build_correctors`
    events       : NanoEvents array for the current chunk
    sample_group : sample ``group`` string (used by ``apply_to`` filtering)
    weights      : :class:`coffea.analysis_tools.Weights` to add corrections to
    """
    for name, (wrapper, cfg) in correctors.items():
        apply_to = cfg.get("apply_to", "all")
        if apply_to != "all" and sample_group not in apply_to:
            continue

        input_arrays = _eval_inputs(cfg, events)
        kind = cfg.get("kind", "event_weight")

        if kind == "event_weight":
            sf = np.asarray(wrapper(*input_arrays), dtype=np.float64)
        elif kind == "object_sf":
            sf = _object_product(wrapper, input_arrays)
        else:
            raise ValueError(f"Unknown correction kind '{kind}' for '{name}'")

        weights.add(name, sf)
