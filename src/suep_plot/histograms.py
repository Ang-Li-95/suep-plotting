"""Build hist.Hist objects from YAML and fill them from coffea NanoEvents.

Histograms are defined in one way: ``histograms.yaml`` lists **regions**, and
each region lists catalog variables (``variables.yaml``) to fill after its
cuts.  :func:`load_histogram_defs` turns that into one definition per
histogram, named ``<region>/<cut>/<variable>``.

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
from dataclasses import dataclass, field

import awkward as ak
import hist
import numpy as np

from pathlib import Path

from .config import GLOBAL_VARIABLES, ConfigError, _merge, load_config_file

# What a region entry may override of its variable's axis.
AXIS_OVERRIDES = ("bins", "lo", "hi", "edges", "label")
# What a catalog variable is made of: the expression plus the axis.
AXIS_KEYS = ("expression",) + AXIS_OVERRIDES
_TRIPLE = ("bins", "lo", "hi")
# Display defaults a catalog variable may carry for its 1D histograms.
DISPLAY_KEYS = ("log_x", "log_y", "flow", "rebin", "blind")
# What a region and one of its object cuts may say (see expand_regions).
REGION_KEYS = ("label", "evt_sel", "evt_vars", "objects")
# Former names of region keys: refused, with the new name in the message.
RENAMED_REGION_KEYS = {"events": "evt_sel", "per_event": "evt_vars"}
CUT_KEYS = ("cut", "label", "vars")
# Keys of the removed ways to define a histogram (written out on its own, or
# as a variant of another): refused in a region entry, with where they went.
REMOVED_ENTRY_KEYS = ("expression", "expression_x", "expression_y",
                      "variants", "keep_base")


class FillError(ValueError):
    """An expression that does not evaluate on a sample: fatal, with its context."""


# ── what the loaders produce ──────────────────────────────────────


@dataclass(frozen=True)
class Axis:
    """What one axis fills and how it is binned: ``edges``, or bins/lo/hi."""

    expression: str
    bins: int | None = None
    lo: float | None = None
    hi: float | None = None
    edges: tuple | None = None
    label: str | None = None

    def __post_init__(self):
        if not self.expression:
            raise ValueError("no expression")
        if self.edges is None and None in (self.bins, self.lo, self.hi):
            raise ValueError("no binning -- needs edges, or bins: [n, lo, hi]")

    def hist_axis(self, name: str, default_label: str) -> hist.axis:
        """The hist axis: Variable for ``edges``, Regular otherwise."""
        label = default_label if self.label is None else self.label
        if self.edges is not None:
            return hist.axis.Variable([float(e) for e in self.edges],
                                      name=name, label=label)
        return hist.axis.Regular(self.bins, self.lo, self.hi, name=name, label=label)

    def to_dict(self, suffix: str = "") -> dict:
        out = {"expression": self.expression}
        if self.edges is not None:
            out["edges"] = list(self.edges)
        else:
            out.update(bins=self.bins, lo=self.lo, hi=self.hi)
        if self.label is not None:
            out["label"] = self.label
        return {f"{k}{suffix}": v for k, v in out.items()}

    @classmethod
    def from_dict(cls, cfg: dict, suffix: str = "") -> "Axis":
        edges = cfg.get(f"edges{suffix}")
        return cls(expression=cfg.get(f"expression{suffix}"),
                   bins=cfg.get(f"bins{suffix}"), lo=cfg.get(f"lo{suffix}"),
                   hi=cfg.get(f"hi{suffix}"),
                   edges=None if edges is None else tuple(edges),
                   label=cfg.get(f"label{suffix}"))


@dataclass(frozen=True)
class Histogram:
    """One histogram: its axes, the cuts before the fill, and how it is drawn.

    ``style`` holds the plot-time keys (``log_y``, ``rebin``, ``flow``,
    ``blind``, ``annotation``, ``log_z``, ...); they do not affect the fill.
    """

    name: str
    x: Axis
    y: Axis | None = None
    selections: tuple = ()
    weight: str | None = None
    style: dict = field(default_factory=dict)

    @property
    def is_2d(self) -> bool:
        return self.y is not None

    @property
    def axes(self) -> tuple:
        return (self.x,) if self.y is None else (self.x, self.y)

    def to_dict(self) -> dict:
        """The flat form stored in the pickles and read by the plots.

        1D: ``expression`` / ``bins`` / ``lo`` / ``hi`` (or ``edges``) /
        ``label``; 2D: the same keys with ``_x`` and ``_y``.  Then the style
        keys, ``selections`` and, if set, ``weight``.
        """
        if self.y is None:
            out = self.x.to_dict()
        else:
            out = {**self.x.to_dict("_x"), **self.y.to_dict("_y")}
        out.update(self.style)
        out["selections"] = list(self.selections)
        if self.weight:
            out["weight"] = self.weight
        return out

    @classmethod
    def from_dict(cls, name: str, cfg: dict) -> "Histogram":
        """A histogram from its flat form (the inverse of :meth:`to_dict`)."""
        two_d = "expression_x" in cfg
        suffixes = ("_x", "_y") if two_d else ("",)
        axis_keys = {f"{k}{s}" for k in AXIS_KEYS for s in suffixes}
        style = {k: v for k, v in cfg.items()
                 if k not in axis_keys | {"selections", "weight", "per_object"}}
        return cls(name, Axis.from_dict(cfg, suffixes[0]),
                   Axis.from_dict(cfg, "_y") if two_d else None,
                   tuple(_as_list(cfg.get("selections"))), cfg.get("weight"), style)


SELECTION_LEVELS = ("event", "object")


@dataclass(frozen=True)
class Selection:
    """One named cut: a boolean expression, per event or per object.

    ``level`` is optional and must agree with the mask (it is read off the
    expression); ``collection`` names what an object-level mask is over, for
    the derived-column steps that select objects with it.
    """

    expression: str
    level: str | None = None
    collection: str | None = None
    label: str | None = None

    def __post_init__(self):
        if not self.expression:
            raise ValueError("no 'expression'")
        if self.level is not None and self.level not in SELECTION_LEVELS:
            raise ValueError(f"'level' must be one of {SELECTION_LEVELS}, "
                             f"got {self.level!r}")

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


def as_histograms(defs: dict) -> dict:
    """``{name: flat dict}`` -> ``{name: Histogram}`` (stored definitions, tests)."""
    return {name: cfg if isinstance(cfg, Histogram) else Histogram.from_dict(name, cfg)
            for name, cfg in defs.items()}


def as_selections(defs: dict, where: str = "selections") -> dict:
    """``{name: mapping}`` -> ``{name: Selection}``, reporting a bad entry by name."""
    out = {}
    for name, cfg in defs.items():
        if isinstance(cfg, Selection):
            out[name] = cfg
            continue
        cfg = dict(cfg or {})
        known = tuple(Selection.__dataclass_fields__)
        unknown = sorted(set(cfg) - set(known))
        try:
            if unknown:
                raise ValueError(f"unknown key(s) {unknown}; known: {', '.join(known)}")
            out[name] = Selection(**{"expression": None, **cfg})
        except ValueError as e:
            raise ConfigError(f"ERROR in {where}: selection '{name}': {e}") from None
    return out


def _bins_list(cfg: dict, where: str) -> dict:
    """``bins: [n, lo, hi]`` -> ``bins`` / ``lo`` / ``hi``."""
    value = cfg.get("bins")
    if not isinstance(value, (list, tuple)):
        return cfg
    if len(value) != 3:
        raise ConfigError(f"ERROR in {where}: bins as a list must be [n, lo, hi]; "
                          "explicit bin edges go under 'edges'")
    clash = [k for k in ("lo", "hi") if k in cfg]
    if clash:
        raise ConfigError(f"ERROR in {where}: bins: [n, lo, hi] together with "
                          f"{clash} -- write the range once")
    n, lo, hi = value
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ConfigError(f"ERROR in {where}: bins: the number of bins must be a "
                          f"positive integer, got {n!r}")
    return {**cfg, "bins": n, "lo": lo, "hi": hi}


def _overlay(base: dict, top: dict) -> dict:
    """*top* over *base*; ``edges`` and bins/lo/hi replace each other."""
    out = dict(base)
    if "edges" in top:
        for k in _TRIPLE:
            out.pop(k, None)
    if any(k in top for k in _TRIPLE):
        out.pop("edges", None)
    out.update(top)
    return out


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


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
        out[name] = _bins_list(cfg, where)
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


def variable_display(variables: dict | None, var) -> dict:
    """The 1D display defaults a catalog variable carries (log_y, flow, ...)."""
    if not variables or var not in variables:
        return {}
    return {k: v for k, v in variables[var].items() if k not in AXIS_KEYS}


# ── regions: event cuts -> object cuts -> variables ───────────────


def normalize_hist_file(defs: dict, path, axes) -> dict:
    """The ``normalize`` hook of :func:`load_histogram_defs` (see load_config_file).

    A per-system family of regions is written once, with a ``_repeat`` inside
    ``regions:`` that uses the file's ``_axes``; it is expanded here, per
    file, so a child set can then override one generated region.
    """
    from .config import _expand_repeats

    out = dict(defs)
    if isinstance(defs.get("regions"), dict):
        regions, _ = _expand_repeats(defs["regions"], path, axes)
        regions.pop("_axes", None)
        out["regions"] = regions
    return out


def _axis_variable(spec, where: str) -> tuple[str, dict]:
    """One axis of an entry -> (catalog variable, overrides of its axis).

    ``name``, or ``{var: name, <overrides>}`` with any of
    :data:`AXIS_OVERRIDES`.
    """
    if isinstance(spec, str):
        return spec, {}
    if not isinstance(spec, dict) or not isinstance(spec.get("var"), str):
        raise ConfigError(f"ERROR in {where}: an axis is a catalog variable name, "
                          "or a mapping {var: <name>, <overrides>}; a new "
                          "expression belongs in variables.yaml")
    unknown = sorted(set(spec) - set(AXIS_OVERRIDES) - {"var"})
    if unknown:
        raise ConfigError(f"ERROR in {where}: unknown axis key(s) {unknown}; "
                          f"known: var, {', '.join(AXIS_OVERRIDES)}")
    return spec["var"], {k: v for k, v in spec.items() if k != "var"}


def _region_entry(spec, where: str) -> tuple[str, list, dict]:
    """One entry of a ``vars`` / ``evt_vars`` list -> (leaf name, axes, rest).

    The forms an entry takes:

    ``name``                              1D, a catalog variable
    ``[x, y]``                            2D, two catalog variables
    ``{var: name, <keys>}``               1D with overrides: binning or label
                                          of the axis, ``selections``,
                                          ``weight``, display keys, ``name``
    ``{x: ..., y: ..., <keys>}``          2D; each axis a variable name or
                                          ``{var: name, <axis overrides>}``

    *axes* is one ``(variable, overrides)`` per axis and *rest* the other
    keys.  The leaf name defaults to the variable (``x_vs_y`` in 2D).
    """
    if isinstance(spec, str):
        return spec, [(spec, {})], {}
    if isinstance(spec, list):
        if len(spec) != 2:
            raise ConfigError(f"ERROR in {where}: a 2D entry is [x, y], got {spec}")
        spec = {"x": spec[0], "y": spec[1]}
    if not isinstance(spec, dict):
        raise ConfigError(f"ERROR in {where}: an entry is a variable name, an "
                          f"[x, y] pair or a mapping, got {spec!r}")
    rest = dict(spec)
    removed = sorted(set(rest) & set(REMOVED_ENTRY_KEYS))
    if removed:
        raise ConfigError(
            f"ERROR in {where}: {removed} cannot be written on a histogram any "
            "more.  An expression and its binning go in variables.yaml (the "
            "entry then names the variable); a variant is an object cut of "
            "the region")
    if "per_object" in rest:
        raise ConfigError(f"ERROR in {where}: 'per_object' is not a key; whether "
                          "a histogram fills per object is read off the "
                          "variable's expression")
    leaf = rest.pop("name", None)

    if "var" in rest:
        if "x" in rest or "y" in rest:
            raise ConfigError(f"ERROR in {where}: give 'var' (1D) or 'x' and 'y' "
                              "(2D), not both")
        overrides = {k: rest.pop(k) for k in AXIS_OVERRIDES if k in rest}
        axes = [_axis_variable({"var": rest.pop("var"), **overrides}, where)]
    elif "x" in rest and "y" in rest:
        stray = sorted(set(rest) & set(AXIS_OVERRIDES))
        if stray:
            raise ConfigError(f"ERROR in {where}: in a 2D entry {stray} go inside "
                              "x: or y:, e.g. x: {var: <name>, bins: [...]}")
        axes = [_axis_variable(rest.pop(ax), f"{where}.{ax}") for ax in ("x", "y")]
    else:
        raise ConfigError(f"ERROR in {where}: an entry needs 'var' (1D) or both "
                          f"'x' and 'y' (2D), got {sorted(spec)}")
    if leaf is None:
        leaf = "_vs_".join(var for var, _ in axes)
    return leaf, axes, rest


def entry_leaf(spec) -> str:
    """The histogram leaf name a region entry produces (see _region_entry)."""
    return _region_entry(spec, "derived_plots.yaml")[0]


def region_hist_name(region: str, cut: str | None, leaf: str) -> str:
    """``<region>/<cut>/<variable>`` (or ``<region>/<variable>`` per event)."""
    return f"{region}/{cut}/{leaf}" if cut else f"{region}/{leaf}"


def _resolved_axis(var: str, overrides: dict, variables: dict, where: str) -> Axis:
    """The catalog variable's axis with an entry's overrides laid over it."""
    if var not in variables:
        close = difflib.get_close_matches(var, list(variables), n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise ConfigError(f"ERROR in {where}: unknown variable '{var}'{hint}")
    base = {k: v for k, v in variables[var].items() if k in AXIS_KEYS}
    try:
        return Axis.from_dict(_overlay(base, _bins_list(overrides, where)))
    except ValueError as e:
        raise ConfigError(f"ERROR in {where}: variable '{var}': {e}") from None


def _entry_histogram(name: str, axes: list, rest: dict, selections: list,
                     annotation: str, variables: dict, where: str) -> Histogram:
    """The histogram of one region entry (see :func:`_region_entry`).

    *selections* are the region's and the cut's; the entry's own come after
    them.  A 1D histogram starts from its variable's display defaults.
    """
    rest = dict(rest)
    selections = tuple(selections) + tuple(_as_list(rest.pop("selections", None)))
    weight = rest.pop("weight", None)
    resolved = [_resolved_axis(var, over, variables, where) for var, over in axes]
    style = variable_display(variables, axes[0][0]) if len(axes) == 1 else {}
    style = {**style, **rest}
    if annotation:
        style.setdefault("annotation", annotation)
    return Histogram(name, *resolved, selections=selections, weight=weight, style=style)


def expand_regions(regions: dict, variables: dict, path="histograms.yaml") -> dict:
    """``regions:`` -> one :class:`Histogram` per (region, cut, variable).

    A region names its event-level cuts (``evt_sel``, AND-ed), the variables to
    fill once per event after them (``evt_vars``), and named object-level
    cuts (``objects``), each with the variables filled after the event cuts
    *and* that cut.  The histograms are named ``<region>/<cut>/<variable>``
    (``<region>/<variable>`` per event), which also becomes their folder in
    the plots.  The region and cut labels go on the figure.
    """
    out: dict = {}

    def add(region, cut, spec, selections, annotation, where):
        leaf, axes, rest = _region_entry(spec, where)
        name = region_hist_name(region, cut, leaf)
        if name in out:
            raise ConfigError(f"ERROR in {where}: '{name}' is listed twice; "
                              "give one of them a 'name:'")
        out[name] = _entry_histogram(name, axes, rest, selections, annotation,
                                     variables, where)

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
        evt_sel = _as_list(region.get("evt_sel"))
        rlabel = region.get("label", rname)

        for spec in region.get("evt_vars") or []:
            add(rname, None, spec, evt_sel, rlabel, f"{where}.evt_vars")

        for alias, cut_cfg in (region.get("objects") or {}).items():
            if cut_cfg is None:
                continue
            cwhere = f"{where}.objects.{alias}"
            if not isinstance(cut_cfg, dict):
                raise ConfigError(f"ERROR in {cwhere}: an object cut is a mapping "
                                  f"of {', '.join(CUT_KEYS)}")
            unknown = sorted(set(cut_cfg) - set(CUT_KEYS))
            if unknown:
                raise ConfigError(f"ERROR in {cwhere}: unknown key(s) {unknown}; "
                                  f"known: {', '.join(CUT_KEYS)}")
            clabel = cut_cfg.get("label",
                                 None if alias == "all" else alias.replace("_", " "))
            annotation = " · ".join(str(t) for t in (rlabel, clabel) if t)
            for spec in cut_cfg.get("vars") or []:
                add(rname, alias, spec, evt_sel + _as_list(cut_cfg.get("cut")),
                    annotation, f"{cwhere}.vars")
    return out


def load_histogram_defs(path: str, variables: dict | None = None) -> dict:
    """A set's histograms.yaml -> ``{name: Histogram}``.

    The file holds ``regions`` and nothing else (see :func:`expand_regions`);
    the variables they list come from the set's catalog (*variables*, by
    default :func:`load_variables` of the set).
    """
    if variables is None:
        variables = load_variables(Path(path).parent)
    raw = load_config_file(path, normalize=normalize_hist_file)
    stray = [k for k in raw if k != "regions"]
    if stray:
        shown = ", ".join(map(str, stray[:5])) + (" ..." if len(stray) > 5 else "")
        raise ConfigError(
            f"ERROR in {path}: histograms are defined inside 'regions' only, "
            f"but the file also has top-level entries: {shown}.\n  Move each "
            "into a region -- 'evt_vars' for a per-event variable, "
            "'objects.<cut>.vars' for a per-object one -- with its expression "
            "and binning in variables.yaml (see docs/configuration.md)")
    return expand_regions(raw.get("regions"), variables, path)


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


def load_selection_defs(path: str) -> dict:
    """A set's selections.yaml -> ``{name: Selection}``."""
    return as_selections(load_config_file(path, normalize=normalize_selection_file),
                         path)


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
