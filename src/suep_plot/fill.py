"""Evaluate config expressions and fill histograms from coffea NanoEvents.

Expressions are written in NanoEvents object syntax and evaluated with these
names in scope:

  * ``events`` — the NanoEvents array (also aliased as ``ev``)
  * ``ak``     — the awkward module
  * ``np``     — numpy
  * a few safe builtins: ``abs``, ``len``, ``min``, ``max``

Examples:  ``events.Jet.pt``,  ``ak.sum(events.Jet.pt, axis=1)``,
``ak.num(events.Muon)``,  ``(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)``.

What is filled is defined in :mod:`suep_plot.histdefs`.
"""

from __future__ import annotations

import functools

import awkward as ak
import hist
import numpy as np

from .histdefs import Histogram, Selection


class FillError(ValueError):
    """An expression that does not evaluate on a sample: fatal, with its context."""


# ── expressions ───────────────────────────────────────────────────


_SAFE_BUILTINS = {"abs": abs, "len": len, "min": min, "max": max}


@functools.lru_cache(maxsize=None)
def compile_expr(expr: str):
    """Compile an expression string into a callable(events) -> array."""
    code = compile(expr, "<config expression>", "eval")

    def _eval(events):
        # One namespace, as eval's *globals*: nested scopes (a lambda, a
        # comprehension) resolve names there, not in eval's locals.
        return eval(  # noqa: S307 - trusted config expressions
            code,
            {"__builtins__": _SAFE_BUILTINS,
             "events": events, "ev": events, "ak": ak, "np": np},
        )

    return _eval


def is_jagged(arr) -> bool:
    """One entry per object (a jagged array) rather than one per event."""
    return isinstance(arr, ak.Array) and arr.ndim > 1


# ── selection columns ─────────────────────────────────────────────


def attach_selection_columns(events, sel_defs: dict, sample: str = "?"):
    """Attach every ``selections.yaml`` mask to *events* as a column of its name.

    After this ``events.<selection>`` is the mask, so an expression can use a
    cut without restating it (``ak.sum(events.electron_sel, axis=1)``).
    Object-level masks stay jagged; missing values become False.

    Called once per chunk, after the derived columns are attached.  A
    selection may read another one's column, in either order.  One that does
    not evaluate is fatal (:class:`FillError`); one named after an existing
    field of ``events`` is skipped, with a message, so it cannot shadow a
    real branch.  Returns *events* with the columns added.
    """
    pending = []
    for name, sel in sel_defs.items():
        if name in events.fields:
            print(f"  (selection '{name}' is already a field of events; "
                  "not attaching it as a column)")
            continue
        pending.append((name, sel.expression))
    # Selections may read each other's columns: retry what failed until a
    # pass makes no progress, then the first remaining failure is real.
    while pending:
        failed = []
        for name, expr in pending:
            try:
                mask = compile_expr(expr)(events)
            except Exception as e:  # noqa: BLE001 - re-raised below if it persists
                failed.append((name, expr, e))
                continue
            if isinstance(mask, ak.Array):
                mask = ak.fill_none(mask, False)
            events = ak.with_field(events, mask, name)
        if len(failed) == len(pending):
            name, expr, e = failed[0]
            raise FillError(f"selection '{name}' ({expr!r}) failed on sample "
                            f"'{sample}': {type(e).__name__}: {e}") from e
        pending = [(n, x) for n, x, _ in failed]
    return events


def prepare_events(events, derive_fn, settings, sel_defs, sample: str = "?"):
    """What every expression sees: derived columns, then selection columns."""
    if derive_fn is not None:
        events = derive_fn(events, settings, sel_defs)
    return attach_selection_columns(events, sel_defs, sample)


# ── filling ───────────────────────────────────────────────────────


def build_histograms(hist_defs: dict, sample_names: list[str]) -> dict[str, hist.Hist]:
    """Create an empty hist.Hist for every histogram, with a dataset axis."""
    histograms = {}
    for name, h in hist_defs.items():
        # The default x title is the histogram's name; a y axis has none.
        axes = [axis.hist_axis(axis_name, default)
                for axis, axis_name, default in zip(h.axes, "xy", (name, ""))]
        histograms[name] = hist.Hist(
            hist.axis.StrCategory(sample_names, name="dataset", growth=True),
            *axes, storage=hist.storage.Weight())
    return histograms


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


def object_selection_on_event_histogram(name: str, selections: list, sample: str) -> str:
    """The error for a per-event histogram that lists a per-object selection.

    An object-level mask says which objects pass, not which events: there is
    no single event-level reading of it (and of two of them over different
    collections, none at all).  The event-level cut is its own selection.
    """
    example = selections[0]
    return (f"histogram '{name}' fills once per event, but selection(s) "
            f"{selections} have one entry per object (sample '{sample}'). "
            "Write the event-level cut as a selection of its own, e.g. "
            f"\"ak.any(events.{example}, axis=1)\", and list that instead")


def same_object_counts(mask, values) -> bool:
    """Whether a per-object *mask* has one entry per object of *values*."""
    n_mask = ak.fill_none(ak.num(mask, axis=1), -1)
    n_values = ak.fill_none(ak.num(values, axis=1), -1)
    return bool(ak.all(n_mask == n_values))


def object_selection_over_other_collection(name: str, selections: list,
                                           sample: str) -> str:
    """The error for an object mask that does not match the histogram's objects."""
    return (f"histogram '{name}': selection(s) {selections} do not have one "
            f"entry per object of the histogram's expression (sample "
            f"'{sample}') -- they select objects of another collection. An "
            "object-level selection can only filter the collection it is "
            "written over")


def _weight_or_one(extra) -> np.ndarray:
    """A histogram's extra per-event weight; missing or non-finite counts as 1.

    The same convention as a reweight applied globally
    (:meth:`reweights.Reweighter.event_weight`), so ``weight: "@map"`` on one
    histogram and the map registered for the whole sample agree.
    """
    if isinstance(extra, ak.Array):
        extra = ak.fill_none(extra, 1.0)
    w = np.asarray(extra, dtype=np.float64)
    return np.where(np.isfinite(w), w, 1.0)


def _per_object(h: Histogram, evaluate, sample: str) -> bool:
    """Whether a histogram fills one entry per object, read off its expression."""
    jagged = {is_jagged(evaluate("histogram", h.name, "expression", axis.expression))
              for axis in h.axes}
    if len(jagged) > 1:
        raise FillError(f"histogram '{h.name}': its x and y expressions must "
                        "both give one entry per object, or both one per event "
                        f"(sample '{sample}')")
    return jagged.pop()


def fill_histograms(
    histograms: dict[str, hist.Hist],
    hist_defs: dict[str, Histogram],
    sel_defs: dict[str, Selection],
    events,
    sample: str,
    weight: np.ndarray | None = None,
    named_weights: dict | None = None,
    layout: dict | None = None,
) -> dict:
    """Fill all histograms from a chunk of NanoEvents.

    Whether a histogram (or a selection) works per object or per event is read
    off its expression: a jagged result is one entry per object.  An
    expression that does not evaluate is fatal (:class:`FillError`), naming
    the histogram, the sample and the expression.

    An object-level selection filters the objects a per-object histogram
    fills.  On a per-event histogram it is an error: the event-level cut
    ("has a passing object") is written as a selection of its own.

    Histogram ``weight`` entries may be NanoEvents expressions or ``"@name"``
    references to a reweight map or a correction (*named_weights*: ``{name:
    events -> weights}``, already resolved for this sample -- see
    :func:`reweights.reweight_weights` and
    :func:`corrections.correction_weights`).  Either may evaluate to a
    jagged (per-object) array: for per-object histograms each object is then
    weighted individually; for event-level histograms the per-object weights
    are multiplied into one weight per event.  A missing or non-finite weight
    counts as 1, as it does for a reweight applied globally.

    *layout*, if given, receives ``{histogram: per_object}`` for every
    histogram filled.  Returns the selection cache (name -> evaluated mask) so
    callers (e.g. the cutflow) can reuse the results without re-evaluating.
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
                if expr[1:] not in (named_weights or {}):
                    raise KeyError(f"unknown reweight map or correction '{expr}'")
                expr_cache[expr] = named_weights[expr[1:]](events)
            else:
                expr_cache[expr] = compile_expr(expr)(events)
        return expr_cache[expr]

    def evaluate(kind: str, name: str, key: str, expr: str):
        try:
            return eval_expr(expr)
        except Exception as e:  # noqa: BLE001 - re-raised with its context
            raise FillError(f"{kind} '{name}' ({key}: {expr!r}) failed on sample "
                            f"'{sample}': {type(e).__name__}: {e}") from e

    sel_cache: dict = {}

    def selection(sel_name: str):
        if sel_name not in sel_cache:
            sel = evaluate("selection", sel_name, "expression",
                           sel_defs[sel_name].expression)
            if isinstance(sel, ak.Array):
                sel = ak.fill_none(sel, False)
            level = sel_defs[sel_name].level
            if level is not None and (level == "object") != is_jagged(sel):
                kind = "object" if is_jagged(sel) else "event"
                raise FillError(f"selection '{sel_name}' is declared 'level: {level}' "
                                f"but its mask has one entry per {kind} (sample "
                                f"'{sample}'); drop the key, it is inferred")
            sel_cache[sel_name] = sel
        return sel_cache[sel_name]

    for name, h in hist_defs.items():
        if name not in histograms:
            continue

        event_mask = np.ones(n_events, dtype=bool)
        obj_masks, obj_sel_names = [], []
        for sel_name in h.selections:
            sel = selection(sel_name)
            if is_jagged(sel):
                obj_masks.append(sel)
                obj_sel_names.append(sel_name)
            else:
                event_mask = event_mask & np.asarray(sel)

        per_object = _per_object(h, evaluate, sample)
        if layout is not None:
            layout[name] = per_object

        if obj_masks and not per_object:
            raise FillError(object_selection_on_event_histogram(
                name, obj_sel_names, sample))

        obj_mask = None
        if obj_masks:
            # A mask filters the objects of the collection it was made over;
            # over another collection it would pair objects by position.
            values = eval_expr(h.x.expression)
            other = [s for s, m in zip(obj_sel_names, obj_masks)
                     if not same_object_counts(m, values)]
            if other:
                raise FillError(object_selection_over_other_collection(
                    name, other, sample))
            obj_mask = obj_masks[0]
            for m in obj_masks[1:]:
                obj_mask = obj_mask & m

        w = weight
        obj_w = None
        if h.weight:
            extra_w = evaluate("histogram", name, "weight", h.weight)
            if isinstance(extra_w, ak.Array) and extra_w.ndim > 1:
                if per_object:
                    obj_w = extra_w  # combined per object inside the fill
                else:
                    w = weight * _weight_or_one(
                        ak.prod(ak.fill_none(extra_w, 1.0), axis=1))
            else:
                w = weight * _weight_or_one(extra_w)

        try:
            _fill(histograms[name], h, eval_expr, sample, event_mask, w,
                  obj_mask, obj_w, per_object)
        except FillError:
            raise
        except Exception as e:  # noqa: BLE001 - re-raised with its context
            raise FillError(f"histogram '{name}' could not be filled on sample "
                            f"'{sample}': {type(e).__name__}: {e}") from e

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
    flat = np.asarray(ak.flatten(ow, axis=None), dtype=np.float64)
    return np.where(np.isfinite(flat), flat, 1.0)   # as _weight_or_one


def _fill(target: hist.Hist, h: Histogram, eval_expr, sample, mask, w,
          obj_mask, obj_w, per_object: bool) -> None:
    """Fill one histogram (1D or 2D) from the events passing *mask*.

    Per object, the values of the objects passing *obj_mask* are flattened and
    each carries its event's weight (times its own *obj_w*, if any); per event
    there is one value per event.  Non-finite values are dropped.
    """
    values = [eval_expr(axis.expression) for axis in h.axes]

    if per_object:
        if obj_mask is not None:
            values = [v[obj_mask] for v in values]
        selected = [_fill_none_safe(v[mask]) for v in values]
        flat = [np.asarray(ak.flatten(v, axis=None)) for v in selected]
        counts = np.asarray(ak.num(selected[0]))
        flat_w = np.repeat(w[mask], counts)
        if obj_w is not None:
            flat_w = flat_w * _flat_obj_weight(obj_w, obj_mask, mask, counts,
                                               h.weight or "")
    else:
        flat = [np.asarray(_fill_none_safe(v[mask])) for v in values]
        flat_w = w[mask]

    flat = [np.asarray(v, dtype=np.float64) for v in flat]
    flat_w = np.asarray(flat_w, dtype=np.float64)

    valid = np.isfinite(flat_w)
    for v in flat:
        valid = valid & np.isfinite(v)
    target.fill(dataset=sample, weight=flat_w[valid],
                **{axis_name: v[valid] for axis_name, v in zip("xy", flat)})
