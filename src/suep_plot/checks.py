"""What can be checked about a config set before it is run.

Static checks need no input file (:func:`validate_config_set`,
:func:`config_warnings`); :func:`check_expressions` evaluates every
expression on the first events of one file.  ``suep-run`` does both before
processing anything, and ``suep-config check`` does nothing else.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from .configset import import_custom, load_columns_config, load_custom_columns
from .corrections import load_correction_defs
from .fill import _compile_expr, prepare_events
from .histdefs import load_histogram_defs, load_selection_defs
from .inputs import load_samples, open_events
from .reweights import build_reweighters, load_reweight_defs


# ── static checks ─────────────────────────────────────────────────


def reference_errors(hist_defs: dict, sel_defs: dict, reweighters: dict) -> list[str]:
    """Names a histogram uses that the set does not define."""
    errors = []
    for hname, h in sorted(hist_defs.items()):
        wspec = h.weight or ""
        if wspec.startswith("@") and wspec[1:] not in reweighters:
            errors.append(f"histogram '{hname}': unknown reweight map '{wspec}' "
                          "(not in reweights.yaml)")
        for sname in h.selections:
            if sname not in sel_defs:
                close = difflib.get_close_matches(sname, list(sel_defs), n=1)
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                errors.append(f"histogram '{hname}': unknown selection "
                              f"'{sname}'{hint}")
    return errors


def custom_config_errors(settings, sel_defs: dict) -> list[str]:
    """What custom/columns.py's ``validate()`` finds wrong with the set, if anything.

    Problems ``derive()`` would otherwise hit only mid-run -- e.g. the
    ``cluster_isolation`` step without the prompt-object selections it reads.
    """
    validate = getattr(import_custom(), "validate", None)
    if validate is None or settings is None:
        return []
    return list(validate(settings, sel_defs))


def validate_config_set(config_dir: str | Path) -> list[str]:
    """Static checks of one config set; returns the problems (empty = fine).

    Loads every file the way ``suep-run`` does and checks what can be checked
    without opening an input file: the YAML itself, that every selection and
    reweight map a histogram names exists, that ``columns.yaml`` is valid and
    that the steps it enables find what they need.
    """
    config_dir = Path(config_dir)
    try:
        sample_defs = load_samples(config_dir / "samples.yaml")
        hist_defs = load_histogram_defs(config_dir / "histograms.yaml")
        sel_defs = load_selection_defs(config_dir / "selections.yaml")
        load_correction_defs(config_dir / "corrections.yaml")
        rw_defs = load_reweight_defs(config_dir / "reweights.yaml")
    except SystemExit as e:
        return [str(e)]

    errors = [] if sample_defs else ["samples.yaml defines no samples"]
    try:
        reweighters = build_reweighters(rw_defs, config_dir)
    except (ValueError, FileNotFoundError, KeyError) as e:
        errors.append(f"reweights.yaml: {e}")
        reweighters = {}
    errors += reference_errors(hist_defs, sel_defs, reweighters)
    try:
        _, settings = load_custom_columns(
            load_columns_config(config_dir / "columns.yaml"), verbose=False)
        errors += custom_config_errors(settings, sel_defs)
    except SystemExit as e:
        errors.append(str(e))
    return errors


def config_warnings(config_dir: str | Path) -> list[str]:
    """Things that work but are probably not what was meant."""
    from .config import CONFIG_FILES, _read, read_set_file
    from .derived import derived_sources, load_derived_plot_defs

    config_dir = Path(config_dir)
    warnings = []
    parent = read_set_file(config_dir).get("extends")
    if parent:
        for name in CONFIG_FILES:
            path = config_dir / name
            if path.exists() and "_extends" not in _read(path):
                warnings.append(
                    f"{name} does not '_extends: {parent}', so it replaces the "
                    f"parent's {name} entirely (set.yaml only supplies files the "
                    "set does not have)")
    try:
        hists = load_histogram_defs(config_dir / "histograms.yaml")
        for name, cfg in load_derived_plot_defs(
                str(config_dir / "derived_plots.yaml")).items():
            needed = derived_sources(name, cfg) or []
            missing = [n for n in needed if n not in hists]
            if missing:
                warnings.append(f"derived plot '{name}' reads {missing}, which this "
                                "set does not fill: it will not be drawn")
    except SystemExit as e:
        warnings.append(str(e))
    return warnings


# ── expressions on real events ────────────────────────────────────


def check_expressions(files, tree, hist_defs, sel_defs, corr_defs, derive_fn=None,
                      reweighters=None, settings=None, sample: str = "?") -> list[str]:
    """Evaluate every expression on the first events of *files*; return the failures.

    Each histogram's and selection's expression must evaluate, and an explicit
    ``level`` must agree with what the expression gives, on every sample -- the fill treats any failure as fatal, so finding them here
    stops the run in seconds instead of on some chunk an hour in.
    """
    from .fill import (is_jagged, object_selection_on_event_histogram,
                       object_selection_over_other_collection, same_object_counts)

    if not files:
        return []
    try:
        events = open_events(files[0], tree)
    except Exception as e:  # noqa: BLE001 - reported
        return [f"cannot open {files[0]}: {type(e).__name__}: {e}"]
    try:
        events = prepare_events(events, derive_fn, settings, sel_defs, sample)
    except Exception as e:  # noqa: BLE001 - reported
        return [f"derived columns / selections failed: {type(e).__name__}: {e}"]

    bad = []

    def evaluate(what: str, expr: str):
        try:
            return _compile_expr(expr)(events)
        except Exception as e:  # noqa: BLE001 - reported
            bad.append(f"{what} ({expr!r}): {type(e).__name__}: {str(e)[:120]}")
            return None

    object_level = {}          # object-level selection -> its mask
    for name, selection in sel_defs.items():
        mask = evaluate(f"selection '{name}'", selection.expression)
        if mask is not None and is_jagged(mask):
            object_level[name] = mask
        level = selection.level
        if mask is not None and level is not None \
                and (level == "object") != is_jagged(mask):
            bad.append(f"selection '{name}': declared 'level: {level}' but the "
                       "mask says otherwise (drop the key, it is inferred)")
    for name, h in hist_defs.items():
        values = [evaluate(f"histogram '{name}' {axis_name}", axis.expression)
                  for axis, axis_name in zip(h.axes, "xy")]
        if all(v is not None for v in values):
            jagged = {is_jagged(v) for v in values}
            per_object_sels = [s for s in h.selections if s in object_level]
            if jagged == {False} and per_object_sels:
                bad.append(object_selection_on_event_histogram(
                    name, per_object_sels, sample))
            if jagged == {True}:
                other = [s for s in per_object_sels
                         if not same_object_counts(object_level[s], values[0])]
                if other:
                    bad.append(object_selection_over_other_collection(
                        name, other, sample))
            if len(jagged) > 1:
                bad.append(f"histogram '{name}': one axis per object, the other "
                           "per event")
        if h.weight and not h.weight.startswith("@"):
            evaluate(f"histogram '{name}' weight", h.weight)
    for name, cfg in corr_defs.items():
        for expr in cfg.get("inputs", []):
            evaluate(f"correction '{name}' input", expr)
    for name, rw in (reweighters or {}).items():
        for expr in rw.exprs:
            evaluate(f"reweight '{name}'", expr)
    return bad
