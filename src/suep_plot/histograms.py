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

from .config import load_config_file


def expand_variants(raw: dict, path: str = "histograms.yaml") -> dict:
    """Expand ``variants:`` into one histogram definition per variant.

    A variant is the same fill under extra selections, so the whole matrix of
    (variable x selection) is written once per variable::

        _variant_sets:                    # reusable; '_' keys are not histograms
          csc_match:
            matched:   {selections: [csc_cluster_matched],   label_prefix: "matched "}
            unmatched: {selections: [csc_cluster_unmatched], label_prefix: "unmatched "}

        csc_cluster_size:
          expression: "events.cscCluster.size"
          per_object: true
          bins: 50
          lo: 0
          hi: 500
          label: 'CSC cluster $N_{\\mathrm{hits}}$'
          variants: csc_match             # or an inline mapping of the same shape

    yields ``csc_cluster_size``, ``csc_cluster_size_matched`` and
    ``csc_cluster_size_unmatched``.  A variant's ``selections`` are *appended*
    to the base ones (base cuts keep applying), ``label_prefix`` /
    ``label_suffix`` decorate the base label, and any other key overrides the
    base outright.  ``keep_base: false`` drops the unselected histogram.
    """
    variant_sets = raw.get("_variant_sets") or {}
    out: dict = {}

    def _fail(msg):
        raise SystemExit(f"ERROR in {path}: {msg}")

    for name, cfg in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(cfg, dict) or "variants" not in cfg:
            out[name] = cfg
            continue

        base = {k: v for k, v in cfg.items() if k not in ("variants", "keep_base")}
        spec = cfg["variants"]
        if isinstance(spec, str):
            if spec not in variant_sets:
                known = ", ".join(sorted(variant_sets)) or "(none defined)"
                _fail(f"histogram '{name}': unknown variant set '{spec}'."
                      f" Defined under _variant_sets: {known}")
            spec = variant_sets[spec]
        if not isinstance(spec, dict) or not spec:
            _fail(f"histogram '{name}': 'variants' must name a variant set or be"
                  " a non-empty mapping of suffix -> overrides")

        if cfg.get("keep_base", True):
            out[name] = base

        for suffix, overrides in spec.items():
            overrides = dict(overrides or {})
            variant = dict(base)

            selections = list(base.get("selections", [])) + \
                list(overrides.pop("selections", []))
            if selections:
                variant["selections"] = selections

            label = overrides.pop("label", variant.get("label"))
            prefix = overrides.pop("label_prefix", "")
            suffix_label = overrides.pop("label_suffix", "")
            if label is not None:
                variant["label"] = f"{prefix}{label}{suffix_label}"

            variant.update(overrides)

            key = f"{name}_{suffix}"
            if key in out or key in raw:
                _fail(f"histogram '{name}': variant '{suffix}' would overwrite"
                      f" the existing histogram '{key}'")
            out[key] = variant

    return out


def load_histogram_defs(path: str) -> dict:
    return expand_variants(load_config_file(path), str(path))


def load_selection_defs(path: str) -> dict:
    return load_config_file(path)


def is_2d(cfg: dict) -> bool:
    return "expression_x" in cfg and "expression_y" in cfg


def _make_axis(cfg: dict, suffix: str, axis_name: str, default_label: str) -> hist.axis:
    """Regular axis from bins/lo/hi, or Variable axis from an ``edges`` list."""
    label_key = "label" + suffix
    edges = cfg.get("edges" + suffix)
    if edges is not None:
        return hist.axis.Variable([float(e) for e in edges],
                                  name=axis_name, label=cfg.get(label_key, default_label))
    return hist.axis.Regular(cfg["bins" + suffix], cfg["lo" + suffix], cfg["hi" + suffix],
                             name=axis_name, label=cfg.get(label_key, default_label))


def build_histograms(hist_defs: dict, sample_names: list[str]) -> dict[str, hist.Hist]:
    """Create empty hist.Hist objects for every histogram × sample."""
    histograms = {}
    for name, cfg in hist_defs.items():
        if is_2d(cfg):
            h = hist.Hist(
                hist.axis.StrCategory(sample_names, name="dataset", growth=True),
                _make_axis(cfg, "_x", "x", name),
                _make_axis(cfg, "_y", "y", ""),
                storage=hist.storage.Weight(),
            )
        else:
            h = hist.Hist(
                hist.axis.StrCategory(sample_names, name="dataset", growth=True),
                _make_axis(cfg, "", "x", name),
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
    reweighters: dict | None = None,
) -> dict:
    """Fill all histograms from a chunk of NanoEvents.

    Histogram ``weight`` entries may be NanoEvents expressions or ``"@name"``
    references to a reweight map (*reweighters*).  Either may evaluate to a
    jagged (per-object) array: for ``per_object`` histograms each object is
    then weighted individually; for event-level histograms the per-object
    weights are multiplied into one weight per event.

    Returns the selection cache (name -> evaluated mask) so callers (e.g. the
    cutflow) can reuse the results without re-evaluating.
    """
    n_events = len(events)
    if weight is None:
        weight = np.ones(n_events, dtype=np.float64)

    # Expression results are cached for the lifetime of the chunk, so an
    # expression shared by several histograms/selections/weights (e.g.
    # ``events.Jet.pt``) is evaluated once.
    expr_cache: dict = {}

    def eval_expr(expr: str):
        if expr not in expr_cache:
            if expr.startswith("@"):
                rw = (reweighters or {}).get(expr[1:])
                if rw is None:
                    raise KeyError(f"unknown reweight map '{expr}'")
                expr_cache[expr] = rw.evaluate(events)
            else:
                expr_cache[expr] = _compile_expr(expr)(events)
        return expr_cache[expr]

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
                sel = eval_expr(sel_cfg["expression"])
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

        w = weight
        obj_w = None
        if cfg.get("weight"):
            extra_w = eval_expr(cfg["weight"])
            if isinstance(extra_w, ak.Array) and extra_w.ndim > 1:
                if is_per_object:
                    obj_w = extra_w  # combined per object inside the fill
                else:
                    w = weight * np.asarray(
                        ak.prod(ak.fill_none(extra_w, 1.0), axis=1), dtype=np.float64)
            else:
                w = weight * np.asarray(_fill_none_safe(extra_w), dtype=np.float64)

        if is_2d(cfg):
            _fill_2d(histograms[name], cfg, eval_expr, sample, event_mask, w,
                     obj_mask, obj_w)
        else:
            _fill_1d(histograms[name], cfg, eval_expr, sample, event_mask, w,
                     obj_mask, obj_w)

    return sel_cache


def _flat_obj_weight(obj_w, obj_mask, mask, counts, wname: str) -> np.ndarray:
    """Flatten a jagged per-object weight, aligned with the flattened values."""
    if obj_mask is not None:
        obj_w = obj_w[obj_mask]
    ow = ak.fill_none(ak.fill_none(obj_w[mask], [], axis=0), 1.0)
    ocounts = np.asarray(ak.num(ow))
    if not np.array_equal(ocounts, counts):
        raise ValueError(
            f"object-level weight '{wname}' does not match the object structure of "
            "this histogram's expression (both must use the same collection)")
    return np.asarray(ak.flatten(ow, axis=None), dtype=np.float64)


def _fill_1d(h, cfg, eval_expr, sample, mask, w, obj_mask=None, obj_w=None):
    try:
        values = eval_expr(cfg["expression"])
    except (KeyError, ValueError, AttributeError):
        return

    if cfg.get("per_object", False):
        if obj_mask is not None:
            values = values[obj_mask]
        selected = _fill_none_safe(values[mask])
        flat_vals = np.asarray(ak.flatten(selected, axis=None))
        counts = np.asarray(ak.num(selected))
        flat_w = np.repeat(w[mask], counts)
        if obj_w is not None:
            flat_w = flat_w * _flat_obj_weight(obj_w, obj_mask, mask, counts,
                                               cfg.get("weight", ""))
    else:
        flat_vals = np.asarray(_fill_none_safe(values[mask]))
        flat_w = w[mask]

    flat_vals = np.asarray(flat_vals, dtype=np.float64)
    flat_w = np.asarray(flat_w, dtype=np.float64)

    valid = np.isfinite(flat_vals) & np.isfinite(flat_w)
    h.fill(dataset=sample, x=flat_vals[valid], weight=flat_w[valid])


def _fill_2d(h, cfg, eval_expr, sample, mask, w, obj_mask=None, obj_w=None):
    try:
        vals_x = eval_expr(cfg["expression_x"])
        vals_y = eval_expr(cfg["expression_y"])
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
        if obj_w is not None:
            flat_w = flat_w * _flat_obj_weight(obj_w, obj_mask, mask, counts,
                                               cfg.get("weight", ""))
    else:
        flat_x = np.asarray(_fill_none_safe(vals_x[mask]))
        flat_y = np.asarray(_fill_none_safe(vals_y[mask]))
        flat_w = w[mask]

    flat_x = np.asarray(flat_x, dtype=np.float64)
    flat_y = np.asarray(flat_y, dtype=np.float64)
    flat_w = np.asarray(flat_w, dtype=np.float64)

    valid = np.isfinite(flat_x) & np.isfinite(flat_y) & np.isfinite(flat_w)
    h.fill(dataset=sample, x=flat_x[valid], y=flat_y[valid], weight=flat_w[valid])
