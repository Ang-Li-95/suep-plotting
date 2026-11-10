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

import difflib
import functools

import awkward as ak
import hist
import numpy as np

from pathlib import Path

from .config import GLOBAL_VARIABLES, ConfigError, _merge, load_config_file

# What one axis is made of -- the keys an ``x:`` / ``y:`` mapping (or a
# catalog variable, see variables.yaml) may set.
AXIS_KEYS = ("expression", "bins", "lo", "hi", "edges", "label")
_TRIPLE = ("bins", "lo", "hi")
# Display defaults a catalog variable may carry for its 1D histograms.
DISPLAY_KEYS = ("log_x", "log_y", "flow", "rebin", "blind")
# What a region and one of its object cuts may say (see expand_regions).
REGION_KEYS = ("label", "evt_sel", "evt_vars", "objects")
# Former names of region keys: refused, with the new name in the message.
RENAMED_REGION_KEYS = {"events": "evt_sel", "per_event": "evt_vars"}
CUT_KEYS = ("cut", "label", "vars")


class FillError(ValueError):
    """An expression that does not evaluate on a sample: fatal, with its context."""


# ── shorthands, normalized per file before the _extends merge ─────


def _bins_list(cfg: dict, suffix: str, where: str) -> dict:
    """``bins: [n, lo, hi]`` -> ``bins`` / ``lo`` / ``hi`` (with the same suffix)."""
    key = f"bins{suffix}"
    value = cfg.get(key)
    if not isinstance(value, (list, tuple)):
        return cfg
    if len(value) != 3:
        raise ConfigError(f"ERROR in {where}: {key} as a list must be [n, lo, hi]; "
                          f"explicit bin edges go under 'edges{suffix}'")
    clash = [f"{k}{suffix}" for k in ("lo", "hi") if f"{k}{suffix}" in cfg]
    if clash:
        raise ConfigError(f"ERROR in {where}: {key}: [n, lo, hi] together with "
                          f"{clash} -- write the range once")
    n, lo, hi = value
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ConfigError(f"ERROR in {where}: {key}: the number of bins must be a "
                          f"positive integer, got {n!r}")
    return {**cfg, key: n, f"lo{suffix}": lo, f"hi{suffix}": hi}


def _normalize_axis(spec, where: str) -> dict:
    """An ``x:`` / ``y:`` value: a variable name, or a mapping of axis keys."""
    if isinstance(spec, str):
        return {"var": spec}
    if not isinstance(spec, dict):
        raise ConfigError(f"ERROR in {where}: an axis is a variable name or a "
                          f"mapping of {list(AXIS_KEYS)}")
    unknown = sorted(set(spec) - set(AXIS_KEYS) - {"var"})
    if unknown:
        raise ConfigError(f"ERROR in {where}: unknown axis key(s) {unknown}; "
                          f"known: var, {', '.join(AXIS_KEYS)}")
    return _bins_list(spec, "", where)


def normalize_hist(cfg, where: str):
    """One histogram definition (or override) in canonical form."""
    if not isinstance(cfg, dict):
        return cfg
    for suffix in ("", "_x", "_y"):
        cfg = _bins_list(cfg, suffix, where)
    for ax in ("x", "y"):
        if cfg.get(ax) is not None:
            cfg = {**cfg, ax: _normalize_axis(cfg[ax], f"{where}.{ax}")}
    # 'selections: name' means the one selection, not its letters.
    if isinstance(cfg.get("selections"), str):
        cfg = {**cfg, "selections": [cfg["selections"]]}
    variants = cfg.get("variants")
    if isinstance(variants, dict):
        cfg = {**cfg, "variants": {v: normalize_hist(o, f"{where}.variants.{v}")
                                   for v, o in variants.items()}}
    return cfg


def normalize_hist_file(defs: dict, path, axes) -> dict:
    """The ``normalize`` hook of :func:`load_histogram_defs` (see load_config_file)."""
    out = {}
    for name, cfg in defs.items():
        if name == "_variant_sets" and isinstance(cfg, dict):
            out[name] = {
                s: {v: normalize_hist(o, f"{path}: _variant_sets.{s}.{v}")
                    for v, o in (variants or {}).items()}
                for s, variants in cfg.items()}
        elif name == "regions" and isinstance(cfg, dict):
            # A per-system family of regions is written once, with the same
            # _repeat and _axes as everything else in the file.
            from .config import _expand_repeats
            regions, _ = _expand_repeats(cfg, path, axes)
            regions.pop("_axes", None)
            out[name] = regions
        elif str(name).startswith("_"):
            out[name] = cfg
        else:
            out[name] = normalize_hist(cfg, f"{path}: histogram '{name}'")
    return out


# ── the variable catalog ──────────────────────────────────────────


def normalize_variable_file(defs: dict, path, axes) -> dict:
    """A catalog entry: an axis (expression, binning, label) + display defaults."""
    out = {}
    for name, cfg in defs.items():
        if str(name).startswith("_") or cfg is None:
            out[name] = cfg
            continue
        where = f"{path}: variable '{name}'"
        if not isinstance(cfg, dict):
            raise ConfigError(f"ERROR in {where}: a variable is a mapping of "
                              f"{', '.join(AXIS_KEYS + DISPLAY_KEYS)}")
        unknown = sorted(set(cfg) - set(AXIS_KEYS) - set(DISPLAY_KEYS))
        if unknown:
            raise ConfigError(f"ERROR in {where}: unknown key(s) {unknown}; known: "
                              f"{', '.join(AXIS_KEYS + DISPLAY_KEYS)}")
        out[name] = _bins_list(cfg, "", where)
    return out


def load_variables(set_dir=None) -> dict:
    """The variable catalog a config set sees.

    The global ``variables.yaml`` (repo root), with the set's own optional
    ``variables.yaml`` merged over it -- an override of one key
    (``llp_decay_r: {bins: [90, 0, 900]}``) keeps the rest of the entry, and
    every histogram that uses the variable follows.
    """
    catalog = load_config_file(GLOBAL_VARIABLES, normalize=normalize_variable_file)
    if set_dir is not None:
        local = load_config_file(Path(set_dir) / "variables.yaml",
                                 normalize=normalize_variable_file)
        catalog = _merge(catalog, local)
    problems = []
    for name, cfg in catalog.items():
        if not cfg.get("expression"):
            problems.append(f"variable '{name}': no expression")
        if "edges" not in cfg and any(k not in cfg for k in _TRIPLE):
            problems.append(f"variable '{name}': no binning (bins: [n, lo, hi] "
                            "or edges)")
    if problems:
        raise ConfigError("ERROR in the variable catalog (variables.yaml):\n  "
                          + "\n  ".join(problems))
    return catalog


# ── regions: event cuts -> object cuts -> variables ───────────────


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _region_entry(spec, where: str) -> tuple[str, dict]:
    """One entry of a ``vars`` / ``evt_vars`` list -> (leaf name, definition).

    ``name`` (a catalog variable, 1D), ``[x, y]`` (2D), or a mapping
    ``{var: name, <overrides>}`` / ``{x: ..., y: ..., <overrides>}`` with an
    optional ``name:`` for the histogram.
    """
    if isinstance(spec, str):
        return spec, {"x": {"var": spec}}
    if isinstance(spec, list):
        if len(spec) != 2:
            raise ConfigError(f"ERROR in {where}: a 2D entry is [x, y], got {spec}")
        x, y = (_normalize_axis(a, where) for a in spec)
        if "var" not in x or "var" not in y:
            raise ConfigError(f"ERROR in {where}: write an inline 2D axis as a "
                              "mapping with a name:  {name: ..., x: ..., y: ...}")
        return f"{x['var']}_vs_{y['var']}", {"x": x, "y": y}
    if not isinstance(spec, dict):
        raise ConfigError(f"ERROR in {where}: an entry is a variable name, an "
                          f"[x, y] pair or a mapping, got {spec!r}")
    spec = dict(spec)
    leaf = spec.pop("name", None)
    var = spec.pop("var", None)
    if var is not None:
        if "x" in spec:
            raise ConfigError(f"ERROR in {where}: give either 'var' or 'x', not both")
        spec["x"] = var
    spec = normalize_hist(spec, where)
    x, y = spec.get("x") or {}, spec.get("y")
    if leaf is None and "var" in x:
        leaf = x["var"] if y is None else (
            f"{x['var']}_vs_{y['var']}" if "var" in y else None)
    if leaf is None:
        raise ConfigError(f"ERROR in {where}: this entry needs a 'name:'")
    return leaf, spec


def region_hist_name(region: str, cut: str | None, leaf: str) -> str:
    """``<region>/<cut>/<variable>`` (or ``<region>/<variable>`` per event)."""
    return f"{region}/{cut}/{leaf}" if cut else f"{region}/{leaf}"


def entry_leaf(spec) -> str:
    """The histogram leaf name a region entry produces (see _region_entry)."""
    return _region_entry(spec, "derived_plots.yaml")[0]


def expand_regions(regions: dict, path="histograms.yaml") -> dict:
    """``regions:`` -> one histogram definition per (region, cut, variable).

    A region names its event-level cuts (``evt_sel``, AND-ed), the variables to
    fill once per event after them (``evt_vars``), and named object-level
    cuts (``objects``), each with the variables filled after the event cuts
    *and* that cut.  The histograms are named ``<region>/<cut>/<variable>``
    (``<region>/<variable>`` per event), which also becomes their folder in
    the plots.  The region and cut labels go on the figure.
    """
    out: dict = {}
    for rname, region in (regions or {}).items():
        if region is None:
            continue
        where = f"{path}: region '{rname}'"
        if not isinstance(region, dict):
            raise ConfigError(f"ERROR in {where}: a region is a mapping of "
                              f"{', '.join(REGION_KEYS)}")
        unknown = sorted(set(region) - set(REGION_KEYS))
        if unknown:
            renamed = [f"'{k}' is now '{RENAMED_REGION_KEYS[k]}'"
                       for k in unknown if k in RENAMED_REGION_KEYS]
            hint = f" ({'; '.join(renamed)})" if renamed else ""
            raise ConfigError(f"ERROR in {where}: unknown key(s) {unknown}{hint}; "
                              f"known: {', '.join(REGION_KEYS)}")
        events = _as_list(region.get("evt_sel"))
        rlabel = region.get("label", rname)

        def add(name, cfg, selections, annotation, entry_where):
            if name in out:
                raise ConfigError(f"ERROR in {entry_where}: '{name}' is listed "
                                  "twice; give one of them a 'name:'")
            cfg = dict(cfg)
            cfg["selections"] = list(selections) + _as_list(cfg.get("selections"))
            if annotation:
                cfg.setdefault("annotation", annotation)
            out[name] = cfg

        for spec in region.get("evt_vars") or []:
            ewhere = f"{where}.evt_vars"
            leaf, cfg = _region_entry(spec, ewhere)
            add(region_hist_name(rname, None, leaf), cfg, events, rlabel, ewhere)

        for alias, cut in (region.get("objects") or {}).items():
            if cut is None:
                continue
            cwhere = f"{where}.objects.{alias}"
            if not isinstance(cut, dict):
                raise ConfigError(f"ERROR in {cwhere}: an object cut is a mapping "
                                  f"of {', '.join(CUT_KEYS)}")
            unknown = sorted(set(cut) - set(CUT_KEYS))
            if unknown:
                raise ConfigError(f"ERROR in {cwhere}: unknown key(s) {unknown}; "
                                  f"known: {', '.join(CUT_KEYS)}")
            clabel = cut.get("label", None if alias == "all" else alias.replace("_", " "))
            annotation = " · ".join(str(t) for t in (rlabel, clabel) if t)
            for spec in cut.get("vars") or []:
                leaf, cfg = _region_entry(spec, f"{cwhere}.vars")
                add(region_hist_name(rname, alias, leaf), cfg,
                    events + _as_list(cut.get("cut")), annotation, cwhere)
    return out


def normalize_selection_file(defs: dict, path, axes) -> dict:
    """A selection may be just its expression: ``name: "<expr>"``."""
    out = {}
    for name, cfg in defs.items():
        if isinstance(cfg, str) and not str(name).startswith("_"):
            cfg = {"expression": cfg}
        elif cfg is not None and not isinstance(cfg, dict) \
                and not str(name).startswith("_"):
            raise ConfigError(f"ERROR in {path}: selection '{name}' must be an "
                              "expression string or a mapping")
        out[name] = cfg
    return out


# ── x: / y: axes -> today's long form, after the merge ────────────


def _overlay(base: dict, top: dict, suffixes=("",)) -> dict:
    """*top* over *base*; per axis, ``edges`` and bins/lo/hi replace each other."""
    out = dict(base)
    for s in suffixes:
        if f"edges{s}" in top:
            for k in _TRIPLE:
                out.pop(f"{k}{s}", None)
        if any(f"{k}{s}" in top for k in _TRIPLE):
            out.pop(f"edges{s}", None)
    out.update(top)
    return out


def _axis(spec: dict, variables: dict | None, where: str) -> dict:
    """One normalized axis mapping, with its variable's defaults filled in."""
    spec = dict(spec)
    var = spec.pop("var", None)
    base = {}
    if var is not None:
        if not variables or var not in variables:
            close = difflib.get_close_matches(var, list(variables or {}), n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ConfigError(f"ERROR in {where}: unknown variable '{var}'{hint}")
        base = {k: v for k, v in variables[var].items() if k in AXIS_KEYS}
    return _overlay(base, spec)


def variable_display(variables: dict | None, var) -> dict:
    """The 1D display defaults a catalog variable carries (log_y, flow, ...)."""
    if not variables or var not in variables:
        return {}
    return {k: v for k, v in variables[var].items() if k not in AXIS_KEYS}


def resolve_axes(name: str, cfg, variables: dict | None = None,
                 path="histograms.yaml"):
    """``x:`` / ``y:`` mappings (and catalog variables) -> the long-form keys.

    1D: the x axis becomes ``expression`` / ``bins`` / ``lo`` / ``hi`` (or
    ``edges``) / ``label``.  2D: the same with ``_x`` / ``_y``.  Keys the
    histogram writes itself in long form take precedence, so an inherited
    ``x: csc_cluster_size`` combines with a local ``hi: 800``.
    """
    if not isinstance(cfg, dict) or ("x" not in cfg and "y" not in cfg):
        return cfg
    where = f"{path}: histogram '{name}'"
    cfg = dict(cfg)
    x_spec, y_spec = cfg.pop("x", None), cfg.pop("y", None)
    if x_spec is None:
        raise ConfigError(f"ERROR in {where}: 'y:' without 'x:'")
    x = _axis(x_spec, variables, f"{where}.x")
    if y_spec is None:
        return _overlay({**variable_display(variables, x_spec.get("var")), **x}, cfg)
    stray = [k for k in ("expression", "bins", "lo", "hi", "edges") if k in cfg]
    if stray:
        raise ConfigError(f"ERROR in {where}: a 2D histogram takes its axes from "
                          f"x: and y:, not {stray}")
    y = _axis(y_spec, variables, f"{where}.y")
    long = {f"{k}_x": v for k, v in x.items()}
    long.update({f"{k}_y": v for k, v in y.items()})
    return _overlay(long, cfg, ("_x", "_y"))


def check_axes(defs: dict, path="histograms.yaml") -> None:
    """Every resolved histogram must say what it fills and how it is binned."""
    problems = []
    for name, cfg in defs.items():
        if not isinstance(cfg, dict):
            problems.append(f"histogram '{name}': not a mapping")
            continue
        suffixes = ("_x", "_y") if is_2d(cfg) else ("",)
        for s in suffixes:
            if not cfg.get(f"expression{s}"):
                problems.append(f"histogram '{name}': no expression{s}")
            has_edges = f"edges{s}" in cfg
            triple = [f"{k}{s}" for k in _TRIPLE if f"{k}{s}" in cfg]
            if has_edges and triple:
                problems.append(
                    f"histogram '{name}': both edges{s} and {triple} -- when "
                    f"replacing inherited edges with bins{s}, add 'edges{s}: null' "
                    f"(or 'bins{s}: null' etc. the other way round)")
            elif not has_edges and len(triple) != 3:
                missing = [f"{k}{s}" for k in _TRIPLE if f"{k}{s}" not in cfg]
                problems.append(f"histogram '{name}': no binning -- needs "
                                f"edges{s} or bins{s}/lo{s}/hi{s} (missing {missing})")
    if problems:
        raise ConfigError(f"ERROR in {path}:\n  " + "\n  ".join(problems))


def _override_axis(variant: dict, ax: str, spec, fail) -> dict:
    """*variant* with its ``x`` / ``y`` axis overridden by a variant's *spec*.

    The base may carry the axis as an ``x:`` / ``y:`` mapping (not resolved
    yet) or in long form.  A spec naming a catalog variable replaces the axis;
    one giving only some keys (``{hi: 800}``) changes those and keeps the rest.
    """
    if isinstance(spec, str):
        spec = {"var": spec}
    if not isinstance(spec, dict):
        fail(f"'{ax}:' must be a variable name or a mapping of axis keys")
    base_axis = variant.get(ax)
    if isinstance(base_axis, dict):
        merged = dict(spec) if "var" in spec else _overlay(base_axis, spec)
        return {**variant, ax: merged}
    two_d = is_2d(variant)
    if "var" in spec or (ax == "y" and not two_d):
        fail(f"'{ax}:' cannot be overridden here because the base histogram is "
             "written in long form (expression / bins / ...); write the base "
             "with 'x:' (and 'y:'), or override the long-form keys directly")
    s = f"_{ax}" if two_d else ""
    return _overlay(variant, {f"{k}{s}": v for k, v in spec.items()}, (s,))


def _decorate_label(cfg):
    """Apply a variant's deferred ``label_prefix`` / ``label_suffix``.

    Deferred when the base label was not known at expansion time -- it comes
    from the catalog variable, filled in by :func:`resolve_axes`.
    """
    if not isinstance(cfg, dict) or "_label_decor" not in cfg:
        return cfg
    cfg = dict(cfg)
    prefix, suffix = cfg.pop("_label_decor")
    if cfg.get("label") is not None:
        cfg["label"] = f"{prefix}{cfg['label']}{suffix}"
    return cfg


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
    ``label_suffix`` decorate the base label, an ``x:`` / ``y:`` is merged
    into the base axis (``x: {hi: 800}`` changes the range, ``x: other_var``
    replaces the axis), and any other key overrides the base outright.
    ``keep_base: false`` drops the unselected histogram.

    Runs on the definitions as written, before :func:`resolve_axes`.
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

            selections = _as_list(base.get("selections")) + \
                _as_list(overrides.pop("selections", None))
            if selections:
                variant["selections"] = selections

            # An axis override is merged into the base axis, never left next
            # to it: the base's own keys would otherwise win when the axes
            # are resolved and the override would be silently ignored.
            for ax in ("x", "y"):
                if overrides.get(ax) is not None:
                    variant = _override_axis(
                        variant, ax, overrides.pop(ax),
                        lambda msg: _fail(f"histogram '{name}' variant "
                                          f"'{suffix}': {msg}"))

            label = overrides.pop("label", variant.get("label"))
            prefix = overrides.pop("label_prefix", "")
            suffix_label = overrides.pop("label_suffix", "")
            if label is not None:
                variant["label"] = f"{prefix}{label}{suffix_label}"
            elif prefix or suffix_label:
                # The label comes from the catalog variable: decorate it once
                # the axes are resolved (see _decorate_label).
                variant["_label_decor"] = (prefix, suffix_label)

            variant.update(overrides)

            key = f"{name}_{suffix}"
            if key in out or key in raw:
                _fail(f"histogram '{name}': variant '{suffix}' would overwrite"
                      f" the existing histogram '{key}'")
            out[key] = variant

    return out


def load_histogram_defs(path: str, variables: dict | None = None) -> dict:
    """A set's histograms.yaml, fully expanded to today's long form.

    Shorthands are normalized per file; ``regions`` expanded into one
    histogram per (region, cut, variable); ``x:`` / ``y:`` axes and catalog
    variables resolved after the merge, against the set's own catalog
    (*variables*, by default :func:`load_variables` of the set);
    ``variants`` expanded; every result checked for an expression and a
    binning.
    """
    if variables is None:
        variables = load_variables(Path(path).parent)
    # with_fragments: expand_variants reads _variant_sets off the result,
    # including ones inherited through _extends.
    raw = load_config_file(path, with_fragments=True, normalize=normalize_hist_file)
    generated = expand_regions(raw.pop("regions", None), path)
    clash = sorted(set(generated) & set(raw))
    if clash:
        raise ConfigError(f"ERROR in {path}: region histogram(s) {clash} are also "
                          "defined on their own")
    raw.update(generated)
    # Variants first, on the axes as written: a variant may override an axis,
    # and that has to be merged into the base axis before it is resolved.
    defs = expand_variants(raw, str(path))
    defs = {name: _decorate_label(resolve_axes(name, cfg, variables, path))
            for name, cfg in defs.items()}
    check_axes(defs, path)
    return defs


def load_selection_defs(path: str) -> dict:
    """A set's selections.yaml; every selection must have an expression."""
    defs = load_config_file(path, normalize=normalize_selection_file)
    missing = [name for name, cfg in defs.items()
               if not isinstance(cfg, dict) or not cfg.get("expression")]
    if missing:
        raise ConfigError(f"ERROR in {path}: selection(s) {missing} have no "
                          "'expression'")
    return defs


def attach_selection_columns(events, sel_defs: dict, sample: str = "?"):
    """Attach every ``selections.yaml`` mask to *events* as a column of its name.

    A selection is a named boolean array, which makes it exactly the kind of
    thing an expression wants to reach: after this, ``electron_sel`` is
    available as ``events.electron_sel``, so a histogram counts what the cut
    keeps without restating it::

        # selections.yaml -- the cut, written once
        electron_sel:
          level: object
          collection: Electron
          expression: "(events.Electron.pt > 10) & (events.Electron.cutBased >= 1)"

        # histograms.yaml -- reaches for the mask, never re-derives it
        n_electron_sel:
          expression: "ak.sum(events.electron_sel, axis=1)"

    Object-level masks arrive jagged, so ``ak.sum(..., axis=1)`` is the
    multiplicity and ``ak.any(..., axis=1)`` the "has one" flag; event-level
    masks are flat and usable directly. Because the column is the selection,
    the plot and the cut cannot drift apart -- the alternative is two copies of
    the same physics, and retuning a working point silently moves only one.

    Called once per chunk, after the derived columns are attached, so masks over
    derived collections (``events.cscCluster.matched``) work like any other.
    A selection may read another one's column whichever order the two are
    written in.

    A selection that does not evaluate is fatal (:class:`FillError`): a mask
    silently missing on one sample is a cut silently not applied.  A selection
    whose name is already a field of ``events`` is skipped, with a message:
    shadowing a real branch (a selection named ``Jet``) would change what
    every other expression means.

    Returns *events* with the columns added.
    """
    pending = []
    for name, cfg in sel_defs.items():
        if not cfg or not cfg.get("expression"):
            continue
        if name in events.fields:
            print(f"  (selection '{name}' is already a field of events; "
                  "not attaching it as a column)")
            continue
        pending.append((name, cfg["expression"]))
    # Selections may read each other's columns: retry what failed until a
    # pass makes no progress, then the first remaining failure is real.
    while pending:
        failed = []
        for name, expr in pending:
            try:
                mask = _compile_expr(expr)(events)
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
        # One namespace, passed as eval's *globals*, deliberately: a name in
        # eval's locals is invisible to a nested scope, so with the two split a
        # lambda or comprehension in a config expression died with
        # "NameError: name 'ak' is not defined".  Globals are what nested
        # scopes resolve against, so this is what makes them work.
        return eval(  # noqa: S307 - trusted config expressions
            code,
            {"__builtins__": _SAFE_BUILTINS,
             "events": events, "ev": events, "ak": ak, "np": np},
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


def is_jagged(arr) -> bool:
    """One entry per object (a jagged array) rather than one per event."""
    return isinstance(arr, ak.Array) and arr.ndim > 1


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


def _per_object(name: str, cfg: dict, evaluate, sample: str) -> bool:
    """Whether a histogram fills one entry per object, read off its expression.

    An explicit ``per_object:`` must agree with the expression: if it does
    not, the old behaviour was an empty histogram or a crash deep in the fill.
    """
    keys = ("expression_x", "expression_y") if is_2d(cfg) else ("expression",)
    jagged = [is_jagged(evaluate("histogram", name, k, cfg[k])) for k in keys]
    if len(set(jagged)) > 1:
        raise FillError(f"histogram '{name}': expression_x and expression_y must "
                        "both give one entry per object, or both one per event "
                        f"(sample '{sample}')")
    declared = cfg.get("per_object")
    if declared is not None and bool(declared) != jagged[0]:
        kind = "object" if jagged[0] else "event"
        raise FillError(f"histogram '{name}' is declared 'per_object: {declared}' "
                        f"but its expression gives one entry per {kind} "
                        f"(sample '{sample}'); drop the key, it is inferred")
    return jagged[0]


def fill_histograms(
    histograms: dict[str, hist.Hist],
    hist_defs: dict,
    sel_defs: dict,
    events,
    sample: str,
    weight: np.ndarray | None = None,
    reweighters: dict | None = None,
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
    references to a reweight map (*reweighters*).  Either may evaluate to a
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
                rw = (reweighters or {}).get(expr[1:])
                if rw is None:
                    raise KeyError(f"unknown reweight map '{expr}'")
                expr_cache[expr] = rw.evaluate(events)
            else:
                expr_cache[expr] = _compile_expr(expr)(events)
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
            sel_cfg = sel_defs[sel_name]
            sel = evaluate("selection", sel_name, "expression", sel_cfg["expression"])
            if isinstance(sel, ak.Array):
                sel = ak.fill_none(sel, False)
            level = sel_cfg.get("level")
            if level is not None and (level == "object") != is_jagged(sel):
                kind = "object" if is_jagged(sel) else "event"
                raise FillError(f"selection '{sel_name}' is declared 'level: {level}' "
                                f"but its mask has one entry per {kind} (sample "
                                f"'{sample}'); drop the key, it is inferred")
            sel_cache[sel_name] = sel
        return sel_cache[sel_name]

    for name, cfg in hist_defs.items():
        if name not in histograms:
            continue

        event_mask = np.ones(n_events, dtype=bool)
        obj_masks, obj_sel_names = [], []
        for sel_name in cfg.get("selections", []):
            sel = selection(sel_name)
            if is_jagged(sel):
                obj_masks.append(sel)
                obj_sel_names.append(sel_name)
            else:
                event_mask = event_mask & np.asarray(sel)

        per_object = _per_object(name, cfg, evaluate, sample)
        if layout is not None:
            layout[name] = per_object

        if obj_masks and not per_object:
            raise FillError(object_selection_on_event_histogram(
                name, obj_sel_names, sample))

        obj_mask = None
        if obj_masks:
            # A mask filters the objects of the collection it was made over;
            # over another collection it would pair objects by position.
            key = "expression_x" if is_2d(cfg) else "expression"
            values = eval_expr(cfg[key])
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
        if cfg.get("weight"):
            extra_w = evaluate("histogram", name, "weight", cfg["weight"])
            if isinstance(extra_w, ak.Array) and extra_w.ndim > 1:
                if per_object:
                    obj_w = extra_w  # combined per object inside the fill
                else:
                    w = weight * _weight_or_one(
                        ak.prod(ak.fill_none(extra_w, 1.0), axis=1))
            else:
                w = weight * _weight_or_one(extra_w)

        try:
            if is_2d(cfg):
                _fill_2d(histograms[name], cfg, eval_expr, sample, event_mask, w,
                         obj_mask, obj_w, per_object)
            else:
                _fill_1d(histograms[name], cfg, eval_expr, sample, event_mask, w,
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


def _fill_1d(h, cfg, eval_expr, sample, mask, w, obj_mask=None, obj_w=None,
             per_object=None):
    values = eval_expr(cfg["expression"])
    if per_object is None:
        per_object = cfg.get("per_object", is_jagged(values))

    if per_object:
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


def _fill_2d(h, cfg, eval_expr, sample, mask, w, obj_mask=None, obj_w=None,
              per_object=None):
    vals_x = eval_expr(cfg["expression_x"])
    vals_y = eval_expr(cfg["expression_y"])
    if per_object is None:
        per_object = cfg.get("per_object", is_jagged(vals_x))

    if per_object:
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
