"""Corrections via coffea's own tooling.

Scale factors are wrapped with :class:`coffea.lookup_tools.correctionlib_wrapper`
(which evaluates correctionlib ``Correction`` objects on awkward arrays) and
combined through :class:`coffea.analysis_tools.Weights`, coffea's standard
event-weight manager (nominal + up/down systematics, weight statistics).

``kind``:
  * ``event_weight`` — one weight per event; added directly to ``Weights``.
  * ``object_sf``     — one SF per object; the per-event weight is the product
    over objects.  Scalar inputs (e.g. a ``"nominal"`` systematic string) are
    passed through unflattened, so the correctionlib string/category axis
    works correctly.

A histogram can also name a correction as its own weight (``weight:
"@<name>"``): a per-object histogram then weights each object with its own
scale factor.  ``fill_only: true`` keeps such a correction out of the event
weight, so it is not applied twice.
"""

from __future__ import annotations

import functools
import os

import awkward as ak
import numpy as np

# Ensure the scipy/_lazywhere shim is applied before coffea.lookup_tools loads.
from . import _compat  # noqa: F401
from .config import load_config_file, name_list, weight_concerns
from .fill import compile_expr


def load_correction_defs(path: str) -> dict:
    return load_config_file(path)


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
        # 'apply_to: background' is the one group, never a substring match.
        cfg = {**cfg, **dict(zip(("samples", "apply_to"), _targets(name, cfg)))}
        cset = correctionlib.CorrectionSet.from_file(_resolve_file(cfg["file"]))
        correctors[name] = (correctionlib_wrapper(cset[cfg["name"]]), cfg)
    return correctors


def _targets(name: str, cfg: dict) -> tuple:
    """(samples, apply_to) of a correction, as lists of names.

    ``apply_to: background`` is the one group, never a substring match.
    """
    where = f"correction '{name}'"
    return (name_list(cfg.get("samples"), where, "samples"),
            name_list(cfg.get("apply_to", "all") or "all", where, "apply_to",
                      keep=("all",)))


def _eval_inputs(cfg, events):
    return [compile_expr(inp)(events) for inp in cfg["inputs"]]


def _scale_factors(wrapper, input_arrays):
    """The correction evaluated with the structure of its inputs.

    Jagged (per-object) inputs are flattened for the evaluator and the result
    is given back per object; scalar inputs (systematic strings, numbers) are
    passed through so the correctionlib category axis is respected.  With no
    jagged input the result is one value per event.
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
        return np.asarray(wrapper(*flat_inputs), dtype=np.float64)
    return ak.unflatten(wrapper(*flat_inputs), counts)


def evaluate_correction(wrapper, cfg: dict, events):
    """One correction on a chunk: per event, or per object for ``object_sf``."""
    input_arrays = _eval_inputs(cfg, events)
    kind = cfg.get("kind", "event_weight")
    if kind == "event_weight":
        return np.asarray(wrapper(*input_arrays), dtype=np.float64)
    if kind == "object_sf":
        return _scale_factors(wrapper, input_arrays)
    raise ValueError(f"unknown correction kind '{kind}' (event_weight or object_sf)")


def concerns(name: str, cfg: dict, sample: str, group: str, is_data: bool) -> bool:
    """Whether a correction is meant for a sample (see config.weight_concerns)."""
    samples, apply_to = _targets(name, cfg)
    return weight_concerns(samples, apply_to, sample, group, is_data)


def apply_corrections(correctors: dict, events, sample: str, group: str,
                      is_data: bool, weights) -> None:
    """Register every applicable correction into a coffea ``Weights`` object.

    A per-object scale factor enters the event weight as its product over the
    event's objects.  A ``fill_only`` correction is skipped: it is used only
    where a histogram names it (see :func:`correction_weights`).

    Parameters
    ----------
    correctors : output of :func:`build_correctors`
    events     : NanoEvents array for the current chunk
    sample, group, is_data : the sample, for the ``samples`` / ``apply_to`` rule
    weights    : :class:`coffea.analysis_tools.Weights` to add corrections to
    """
    for name, (wrapper, cfg) in correctors.items():
        if cfg.get("fill_only") or not concerns(name, cfg, sample, group, is_data):
            continue
        try:
            sf = evaluate_correction(wrapper, cfg, events)
        except ValueError as e:
            raise ValueError(f"correction '{name}': {e}") from None
        if isinstance(sf, ak.Array):
            sf = np.asarray(ak.fill_none(ak.prod(sf, axis=1), 1.0), dtype=np.float64)
        weights.add(name, sf)


def _no_correction(events):
    return 1.0


def correction_weights(correctors: dict, sample: str, group: str,
                       is_data: bool) -> dict:
    """``{name: events -> weights}`` for a histogram's ``weight: "@name"``.

    For a sample the correction is meant for, the weights are the correction
    as evaluated (per object for ``object_sf``); for the others they are 1, so
    a histogram that names a scale factor can be filled for every sample of
    the set.
    """
    return {name: (functools.partial(evaluate_correction, wrapper, cfg)
                   if concerns(name, cfg, sample, group, is_data) else _no_correction)
            for name, (wrapper, cfg) in correctors.items()}
