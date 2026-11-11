"""What can be checked about a config set before it is run.

Static checks need no input file (:func:`validate_config_set`,
:func:`config_warnings`); :func:`check_expressions` evaluates every
expression on the first events of one file.  ``suep-run`` does both before
processing anything, and ``suep-config check`` does nothing else.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from .configset import ConfigSet, load_config_set
from .fill import compile_expr, prepare_events
from .histdefs import load_histogram_defs
from .inputs import open_events


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


def custom_config_errors(cs: ConfigSet) -> list[str]:
    """What custom/columns.py's ``validate()`` finds wrong with the set, if anything.

    Problems ``derive()`` would otherwise hit only mid-run -- e.g. the
    ``cluster_isolation`` step without the prompt-object selections it reads.
    """
    if cs.validate_fn is None or cs.settings is None:
        return []
    return list(cs.validate_fn(cs.settings, cs.selections))


def sample_errors(samples: dict) -> list[str]:
    """Samples that name no input: a misspelled name next to an override.

    ``dy_2mu_50to12: {max_files: 10}`` (the registry has ``dy_2mu_50to120``)
    is a sample of its own with nothing to read, not an override.
    """
    return [f"sample '{name}': no 'files' -- not in the registry and not "
            "defined here (a misspelled name?)"
            for name, cfg in samples.items() if not (cfg or {}).get("files")]


def static_errors(cs: ConfigSet) -> list[str]:
    """What is wrong with a loaded set, as far as it shows without input files."""
    return sample_errors(cs.samples) \
        + reference_errors(cs.histograms, cs.selections, cs.reweighters) \
        + custom_config_errors(cs)


def load_and_validate(config_dir: str | Path) -> tuple[ConfigSet | None, list[str]]:
    """Load a set and run the static checks -> (the set, its problems).

    The set is ``None`` when a file of it does not load at all; otherwise
    every problem found is listed, not just the first.
    """
    problems: list[str] = []
    try:
        cs = load_config_set(config_dir, problems)
    except SystemExit as e:
        return None, [str(e)]
    if not cs.samples:
        problems.append("samples.yaml defines no samples")
    return cs, problems + static_errors(cs)


def validate_config_set(config_dir: str | Path) -> list[str]:
    """Static checks of one config set; returns the problems (empty = fine).

    Loads every file the way ``suep-run`` does and checks what can be checked
    without opening an input file: the YAML itself, that every selection and
    reweight map a histogram names exists, that ``columns.yaml`` is valid and
    that the steps it enables find what they need.
    """
    return load_and_validate(config_dir)[1]


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
                      reweighters=None, settings=None, sample: str = "?",
                      is_data: bool = False) -> list[str]:
    """Evaluate every expression on the first events of *files*; return the failures.

    Each histogram's and selection's expression must evaluate, and an explicit
    ``level`` must agree with what the expression gives, on every sample -- the
    fill treats any failure as fatal, so finding them here stops the run in
    seconds instead of on some chunk an hour in.  For a data sample
    (*is_data*) the ``mc_only`` histograms are left out, as they are in the
    fill.
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
            return compile_expr(expr)(events)
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
        if is_data and h.mc_only:
            continue
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
