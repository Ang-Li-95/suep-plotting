"""Build hist.Hist objects from YAML and fill them from coffea NanoEvents.

Expressions are written in NanoEvents object syntax and evaluated with these
names in scope:

  * ``events`` — the NanoEvents array (also aliased as ``ev``)
  * ``ak``     — the awkward module
  * ``np``     — numpy
  * a few safe builtins: ``abs``, ``len``, ``min``, ``max``

Examples:  ``events.Jet.pt``,  ``ak.sum(events.Jet.pt, axis=1)``,
``ak.num(events.Muon)``,  ``(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)``.
"""

from __future__ import annotations

import functools

import awkward as ak
import hist
import numpy as np
import yaml


def load_histogram_defs(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_selection_defs(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def is_2d(cfg: dict) -> bool:
    return "expression_x" in cfg and "expression_y" in cfg


def build_histograms(hist_defs: dict, sample_names: list[str]) -> dict[str, hist.Hist]:
    """Create empty hist.Hist objects for every histogram × sample."""
    histograms = {}
    for name, cfg in hist_defs.items():
        if is_2d(cfg):
            h = hist.Hist(
                hist.axis.StrCategory(sample_names, name="dataset", growth=True),
                hist.axis.Regular(cfg["bins_x"], cfg["lo_x"], cfg["hi_x"],
                                  name="x", label=cfg.get("label_x", name)),
                hist.axis.Regular(cfg["bins_y"], cfg["lo_y"], cfg["hi_y"],
                                  name="y", label=cfg.get("label_y", "")),
                storage=hist.storage.Weight(),
            )
        else:
            h = hist.Hist(
                hist.axis.StrCategory(sample_names, name="dataset", growth=True),
                hist.axis.Regular(cfg["bins"], cfg["lo"], cfg["hi"],
                                  name="x", label=cfg.get("label", name)),
                storage=hist.storage.Weight(),
            )
        histograms[name] = h
    return histograms


_SAFE_BUILTINS = {"abs": abs, "len": len, "min": min, "max": max}


@functools.lru_cache(maxsize=None)
def _compile_expr(expr: str):
    """Compile an expression string into a callable(events) -> array."""
    code = compile(expr, "<config expression>", "eval")

    def _eval(events):
        return eval(  # noqa: S307 - trusted config expressions
            code,
            {"__builtins__": _SAFE_BUILTINS},
            {"events": events, "ev": events, "ak": ak, "np": np},
        )

    return _eval


def _fill_none_safe(arr):
    """Make an expression result safe against missing values.

    Expressions like ``ak.firsts(...)`` or ``.nearest(...)`` yield ``None``
    entries.  ``None`` events become empty lists (jagged case) and ``None``
    values become NaN, which the finite-value filter in the fill functions
    drops while keeping weights aligned.
    """
    if not isinstance(arr, ak.Array):
        return arr
    if arr.ndim > 1:
        arr = ak.fill_none(arr, [], axis=0)
    return ak.fill_none(arr, np.nan)


def fill_histograms(
    histograms: dict[str, hist.Hist],
    hist_defs: dict,
    sel_defs: dict,
    events,
    sample: str,
    weight: np.ndarray | None = None,
):
    """Fill all histograms from a chunk of NanoEvents."""
    n_events = len(events)
    if weight is None:
        weight = np.ones(n_events, dtype=np.float64)

    sel_cache: dict = {}

    for name, cfg in hist_defs.items():
        if name not in histograms:
            continue

        event_mask = np.ones(n_events, dtype=bool)
        obj_masks = []
        is_per_object = cfg.get("per_object", False)

        for sel_name in cfg.get("selections", []):
            sel_cfg = sel_defs[sel_name]
            level = sel_cfg.get("level", "event")

            if sel_name not in sel_cache:
                sel = _compile_expr(sel_cfg["expression"])(events)
                if isinstance(sel, ak.Array):
                    sel = ak.fill_none(sel, False)
                sel_cache[sel_name] = sel

            if level == "object":
                obj_masks.append(sel_cache[sel_name])
            else:
                event_mask = event_mask & np.asarray(sel_cache[sel_name])

        obj_mask = None
        if obj_masks:
            obj_mask = obj_masks[0]
            for m in obj_masks[1:]:
                obj_mask = obj_mask & m
            if not is_per_object:
                event_mask = event_mask & np.asarray(ak.any(obj_mask, axis=1))
                obj_mask = None

        w = weight.copy()
        if "weight" in cfg and cfg["weight"]:
            extra_w = _fill_none_safe(_compile_expr(cfg["weight"])(events))
            w = w * np.asarray(extra_w)

        if is_2d(cfg):
            _fill_2d(histograms[name], cfg, events, sample, event_mask, w, obj_mask)
        else:
            _fill_1d(histograms[name], cfg, events, sample, event_mask, w, obj_mask)


def _fill_1d(h, cfg, events, sample, mask, w, obj_mask=None):
    try:
        values = _compile_expr(cfg["expression"])(events)
    except (KeyError, ValueError, AttributeError):
        return

    if cfg.get("per_object", False):
        if obj_mask is not None:
            values = values[obj_mask]
        selected = _fill_none_safe(values[mask])
        flat_vals = np.asarray(ak.flatten(selected, axis=None))
        counts = np.asarray(ak.num(selected))
        flat_w = np.repeat(w[mask], counts)
    else:
        flat_vals = np.asarray(_fill_none_safe(values[mask]))
        flat_w = w[mask]

    flat_vals = np.asarray(flat_vals, dtype=np.float64)
    flat_w = np.asarray(flat_w, dtype=np.float64)

    valid = np.isfinite(flat_vals) & np.isfinite(flat_w)
    h.fill(dataset=sample, x=flat_vals[valid], weight=flat_w[valid])


def _fill_2d(h, cfg, events, sample, mask, w, obj_mask=None):
    try:
        vals_x = _compile_expr(cfg["expression_x"])(events)
        vals_y = _compile_expr(cfg["expression_y"])(events)
    except (KeyError, ValueError, AttributeError):
        return

    if cfg.get("per_object", False):
        if obj_mask is not None:
            vals_x = vals_x[obj_mask]
            vals_y = vals_y[obj_mask]
        sel_x = _fill_none_safe(vals_x[mask])
        sel_y = _fill_none_safe(vals_y[mask])
        flat_x = np.asarray(ak.flatten(sel_x, axis=None))
        flat_y = np.asarray(ak.flatten(sel_y, axis=None))
        counts = np.asarray(ak.num(sel_x))
        flat_w = np.repeat(w[mask], counts)
    else:
        flat_x = np.asarray(_fill_none_safe(vals_x[mask]))
        flat_y = np.asarray(_fill_none_safe(vals_y[mask]))
        flat_w = w[mask]

    flat_x = np.asarray(flat_x, dtype=np.float64)
    flat_y = np.asarray(flat_y, dtype=np.float64)
    flat_w = np.asarray(flat_w, dtype=np.float64)

    valid = np.isfinite(flat_x) & np.isfinite(flat_y) & np.isfinite(flat_w)
    h.fill(dataset=sample, x=flat_x[valid], y=flat_y[valid], weight=flat_w[valid])
