"""The one YAML reader: ``_extends``, ``_include`` and ``_repeat``.

Every config file in a config set is loaded through :func:`load_config_file`,
so the three directives below mean the same thing wherever they appear —
``histograms.yaml``, ``selections.yaml``, ``samples.yaml``, ``columns.yaml``,
``corrections.yaml``, ``reweights.yaml`` and ``derived_plots.yaml`` alike.

``_extends: ../common`` (or a list, ``[../common, ../common_gen]``)
    Inherit the same-named file from another config set.  This file's keys are
    merged over the inherited ones (deeply, for nested mappings), and
    ``some_name: null`` *deletes* an inherited definition -- or, nested,
    removes one key of it (``some_hist: {variants: null}``).  With a list the
    bases are merged left to right and this file merges over the result, so a
    study can compose ``common`` with ``common_gen`` instead of chaining.
    Chains are followed too, and a cycle is an error.

``_include: [../../datasets.yaml, more.yaml]``
    Merge in definitions from shared fragment files.  Unlike ``_extends``,
    which follows the config set, an include names one file directly.

``_registry: ../../datasets.yaml``
    A *pool to select from* rather than definitions to merge — what
    ``samples.yaml`` does with the central dataset registry.  A name listed in
    the file with no value takes the pool entry verbatim, ``name: {...}``
    takes it with those keys overridden, and ``name: {_from: other, ...}``
    bases it on pool entry ``other`` so one dataset can appear under several
    config names.  A file with no ``_registry`` needs every definition spelled
    out, as before.  (Kept distinct from ``_include`` on purpose: selecting
    from a pool and merging definitions want opposite meanings for an empty
    value.)

``_repeat:``
    Expand one template over a list of substitutions — the per-system
    (csc/dt/rpc) loop that most of these files are made of::

        _axes:
          systems: [{SYS: csc, SYSU: CSC}, {SYS: dt, SYSU: DT}]

        _repeat:
          - over: systems              # or an inline list of mappings
            defs:
              llp_in_<SYS>:
                level: object
                expression: "events.llp.in<SYSU>"

    ``<NAME>`` is substituted into keys and into every string leaf.  The sigil
    is angle brackets, not braces, because the labels are full of LaTeX
    (``$N_{\\mathrm{hits}}$``).  A value that is *exactly* one placeholder takes
    the substitution's type rather than its text, so ``hi: <HI>`` with
    ``HI: 500`` stays the integer 500 and ``edges: <EDGES>`` can carry a whole
    list; a placeholder embedded in surrounding text always interpolates as a
    string.  A loop that covers only some systems just lists those; there are
    no conditionals.

Keys starting with ``_`` are directives or reusable fragments, never
definitions — the convention the configs already used.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

DIRECTIVES = ("_extends", "_include", "_registry", "_axes", "_repeat")

_SUBST = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


class ConfigError(SystemExit):
    """A malformed config: reported with the file it came from, not a traceback."""


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"ERROR in {path}: expected a mapping at the top level, "
                          f"got {type(raw).__name__}")
    return raw


def _merge(base: dict, over: dict) -> dict:
    """*over* merged onto *base*; ``None`` deletes, nested mappings merge deeply."""
    out = dict(base)
    for key, value in over.items():
        if value is None and key in out:
            del out[key]
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _substitute(node, values: dict):
    """Replace ``<NAME>`` throughout a template with *values*."""

    def lookup(name: str):
        if name not in values:
            raise ConfigError(
                f"ERROR: '<{name}>' has no value in this _repeat entry; "
                f"it defines {sorted(values)}")
        return values[name]

    def one(text: str):
        # A value that is nothing but a placeholder keeps the substituted
        # type: 'bins: <BINS>' must stay an int, 'edges: <EDGES>' a list.
        whole = _SUBST.fullmatch(text)
        if whole is not None:
            return lookup(whole.group(1))
        return _SUBST.sub(lambda m: str(lookup(m.group(1))), text)

    if isinstance(node, str):
        return one(node)
    if isinstance(node, dict):
        return {str(one(k)) if isinstance(k, str) else k: _substitute(v, values)
                for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    return node


def _expand_repeats(raw: dict, path: Path, inherited_axes: dict):
    """Replace the ``_repeat`` blocks with the definitions they generate.

    Returns ``(defs, axes)``.  Axes are inherited: a base set can define the
    systems axis once and everything extending it can loop over it.
    """
    blocks = raw.get("_repeat")
    axes = {**inherited_axes, **(raw.get("_axes") or {})}
    if not blocks:
        return {k: v for k, v in raw.items() if k != "_repeat"}, axes
    if not isinstance(blocks, list):
        raise ConfigError(f"ERROR in {path}: _repeat must be a list of "
                          "{over, defs} blocks")

    out = {k: v for k, v in raw.items() if k != "_repeat"}

    for i, block in enumerate(blocks):
        if not isinstance(block, dict) or "over" not in block or "defs" not in block:
            raise ConfigError(f"ERROR in {path}: _repeat[{i}] needs both 'over' "
                              "and 'defs'")
        over = block["over"]
        if isinstance(over, str):
            if over not in axes:
                known = ", ".join(sorted(axes)) or "(none defined)"
                raise ConfigError(f"ERROR in {path}: _repeat[{i}] over '{over}' is "
                                  f"not defined under _axes. Defined: {known}")
            over = axes[over]
        if not isinstance(over, list) or not all(isinstance(v, dict) for v in over):
            raise ConfigError(f"ERROR in {path}: _repeat[{i}] 'over' must name an "
                              "_axes entry or be a list of substitution mappings")

        for values in over:
            for name, cfg in _substitute(block["defs"], values).items():
                if name in out:
                    raise ConfigError(
                        f"ERROR in {path}: _repeat[{i}] generates '{name}', which "
                        "this file already defines")
                out[name] = cfg
    return out, axes


def _resolve_registry(defs: dict, registry: dict, path: Path) -> dict:
    """Resolve each entry against the ``_registry`` pool (see the module docs)."""
    out = {}
    for name, cfg in defs.items():
        cfg = dict(cfg or {})
        base_name = cfg.pop("_from", name)
        base = registry.get(base_name)
        if base is None:
            if not cfg:
                known = ", ".join(sorted(registry)) or "(empty registry)"
                raise ConfigError(
                    f"ERROR in {path}: '{name}' is not defined here and has no "
                    f"registry entry '{base_name}'. Available: {known}")
            out[name] = cfg
        else:
            out[name] = {**base, **cfg}
    return out


def load_config_file(path: str | Path, *, with_fragments=False, sources=None,
                     _seen=None) -> dict:
    """Load one config file with ``_extends`` / ``_include`` / ``_repeat`` applied.

    A missing file loads as ``{}``, so an optional config (``columns.yaml``,
    ``reweights.yaml``) needs no special case at the call site.  Directive keys
    are consumed here; what comes back is definitions only.

    What comes back is definitions only: ``_``-prefixed keys are directives or
    reusable fragments, never definitions, so they are stripped.  Pass
    *with_fragments* to keep them -- ``load_histogram_defs`` does, because
    ``expand_variants`` reads ``_variant_sets`` off the merged result.

    Pass a set as *sources* to collect every file that was actually read,
    including the ones reached through ``_extends`` / ``_include`` /
    ``_registry``.  :func:`config_sources` uses it so the up-to-date check can
    see a config a set only inherits -- otherwise editing a shared fragment
    leaves every run that includes it silently "up to date".
    """
    path = Path(path)
    raw = _read(path)
    if sources is not None and path.exists():
        sources.add(path.resolve())

    _seen = _seen or []
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in [*_seen, resolved])
        raise ConfigError(f"ERROR: circular _extends: {chain}")

    merged: dict = {}
    axes: dict = {}

    parents = raw.pop("_extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    for parent in parents:
        base_dir = Path(parent) if Path(parent).is_absolute() else path.parent / parent
        base_path = base_dir / path.name
        if not base_dir.is_dir():
            raise ConfigError(f"ERROR in {path}: _extends config set not found: "
                              f"{base_dir}")
        merged = _merge(merged, load_config_file(
            base_path, with_fragments=True, sources=sources, _seen=[*_seen, resolved]))
        axes.update(merged.pop("_axes", {}) or {})

    includes = raw.pop("_include", [])
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        inc_path = Path(inc) if Path(inc).is_absolute() else path.parent / inc
        if not inc_path.exists():
            raise ConfigError(f"ERROR in {path}: _include file not found: {inc_path}")
        merged = _merge(merged, load_config_file(
            inc_path, with_fragments=True, sources=sources, _seen=[*_seen, resolved]))
        axes.update(merged.pop("_axes", {}) or {})

    registry_spec = raw.pop("_registry", None)

    defs, axes = _expand_repeats(raw, path, axes)
    if axes:
        defs["_axes"] = axes
    if registry_spec is not None:
        specs = [registry_spec] if isinstance(registry_spec, str) else registry_spec
        registry: dict = {}
        for spec in specs:
            reg_path = Path(spec) if Path(spec).is_absolute() else path.parent / spec
            if not reg_path.exists():
                raise ConfigError(f"ERROR in {path}: _registry file not found: "
                                  f"{reg_path}")
            registry.update(load_config_file(reg_path, sources=sources,
                                             _seen=[*_seen, resolved]))
        defs = _resolve_registry(defs, registry, path)

    out = _merge(merged, defs)
    if not with_fragments:
        out = {k: v for k, v in out.items() if not str(k).startswith("_")}
    return out


# The files a config set is made of.
CONFIG_FILES = ("samples.yaml", "histograms.yaml", "selections.yaml",
                "corrections.yaml", "reweights.yaml", "derived_plots.yaml",
                "columns.yaml")

# The subset that decides what lands in the pickles.  derived_plots.yaml is
# deliberately absent: it is read by suep-plot only, so editing a profile or an
# efficiency must not force every sample to be reprocessed.
FILL_CONFIG_FILES = tuple(f for f in CONFIG_FILES if f != "derived_plots.yaml")


def config_sources(config_dir: str | Path, files=CONFIG_FILES) -> set:
    """Every YAML file a config set actually reads, directives followed.

    A set that inherits half its definitions from elsewhere -- ``_extends`` on a
    reference set, ``_include`` of a shared fragment, ``_registry`` on
    datasets.yaml -- is not described by the seven files in its own directory.
    The up-to-date check needs the real list, or editing a shared cut leaves
    every run that includes it reporting "up to date" and plotting stale
    histograms.
    """
    config_dir = Path(config_dir)
    sources: set = set()
    for name in files:
        try:
            load_config_file(config_dir / name, sources=sources)
        except SystemExit:
            # A broken config is the run's problem, not the freshness check's;
            # whatever was read before it failed still counts.
            pass
    return sources
