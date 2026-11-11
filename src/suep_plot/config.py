"""The one YAML reader: ``_extends``, ``_include``, ``_registry``, ``_repeat``.

Every config file of a set is loaded through :func:`load_config_file`, so the
directives mean the same thing in every file:

``_extends: ../common``    inherit the same-named file of another set (or of a
                           list of sets, merged left to right); this file's
                           keys merge over it, and ``name: null`` deletes one
``_include: [a.yaml]``     merge in the definitions of the named files
``_registry: ../x.yaml``   a pool to select from: a name with no value takes
                           the pool entry, ``{_from: other}`` another's
``_axes`` / ``_repeat``    expand one template over a list of substitutions
                           (``<NAME>`` in keys and string values)
``set.yaml``               ``extends:`` supplies every file the set does not
                           have; also ``description:`` and ``abstract:``

A key starting with ``_`` is a directive, never a definition, and an unknown
one is an error.  docs/configuration.md is the reference for all of this;
docs/design-notes.md says why it is built this way.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import yaml

DIRECTIVES = ("_extends", "_include", "_registry", "_axes", "_repeat")

# The checkout: configs/, custom/, variables.yaml and datasets.yaml live here,
# next to src/.
REPO_ROOT = Path(__file__).resolve().parents[2]

SET_FILE = "set.yaml"
SET_KEYS = ("extends", "description", "abstract")

_SUBST = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


class ConfigError(SystemExit):
    """A malformed config: reported with the file it came from, not a traceback."""


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"ERROR in {path}: not valid YAML:\n{e}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"ERROR in {path}: expected a mapping at the top level, "
                          f"got {type(raw).__name__}")
    return raw


def _merge(base: dict, over: dict) -> dict:
    """*over* merged onto *base*; ``None`` deletes, nested mappings merge deeply.

    A ``None`` for a key the base does not have is dropped too, never stored:
    there is nothing to delete, and a stored ``None`` only resurfaces later as
    an unrelated crash.
    """
    out = dict(base)
    for key, value in over.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def name_list(value, where: str, key: str, keep=()):
    """A config value naming groups or samples -> a list of names.

    A bare string is one name, not a sequence of letters: ``apply_to:
    background`` must not match by substring (``"" in "background"`` is true,
    so a sample without a group would be picked up).  Values listed in *keep*
    (``"all"``, ``None``) pass through unchanged.
    """
    if value is None or (isinstance(value, str) and value in keep):
        return value
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"{where}: '{key}' must be a name or a list of names, "
                     f"got {value!r}")


def read_set_file(config_dir: str | Path) -> dict:
    """A config set's ``set.yaml`` (``{}`` if it has none), checked."""
    path = Path(config_dir) / SET_FILE
    meta = _read(path)
    unknown = sorted(set(meta) - set(SET_KEYS))
    if unknown:
        raise ConfigError(f"ERROR in {path}: unknown key(s) {unknown}; "
                          f"known: {list(SET_KEYS)}")
    return meta


def _check_directives(raw: dict, path: Path) -> None:
    known = DIRECTIVES
    unknown = [k for k in raw if isinstance(k, str) and k.startswith("_")
               and k not in known]
    if unknown:
        hints = [f"'{k}' -> did you mean '{c[0]}'?" for k in unknown
                 if (c := difflib.get_close_matches(k, known, n=1))]
        hint = ("\n  " + "\n  ".join(hints)) if hints else ""
        raise ConfigError(f"ERROR in {path}: unknown directive(s) {unknown}; "
                          f"known: {', '.join(known)}{hint}")


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
        explicit = "_from" in cfg
        base_name = cfg.pop("_from", name)
        base = registry.get(base_name)
        if base is None:
            # An entry with keys of its own may be a sample defined right
            # here -- but not one that says which registry entry it is.
            if explicit or not cfg:
                known = ", ".join(sorted(registry)) or "(empty registry)"
                raise ConfigError(
                    f"ERROR in {path}: '{name}' is not defined here and has no "
                    f"registry entry '{base_name}'. Available: {known}")
            out[name] = cfg
        else:
            out[name] = {**base, **cfg}
    return out


def load_config_file(path: str | Path, *, with_fragments=False, sources=None,
                     normalize=None, origins=None, _seen=None) -> dict:
    """Load one config file with ``_extends`` / ``_include`` / ``_repeat`` applied.

    A missing file loads as ``{}`` -- or, when its set has a ``set.yaml`` with
    ``extends:``, as that parent set's file -- so an optional config
    (``columns.yaml``, ``reweights.yaml``) needs no special case at the call
    site.

    What comes back is definitions only: ``_``-prefixed keys are directives,
    never definitions, so they are stripped.  *with_fragments* keeps them; the
    loader uses it on the files a file inherits from, to carry their ``_axes``
    down.

    *normalize* ``(defs, path, axes) -> defs`` rewrites each file's own
    definitions into canonical form *before* they are merged over what the file
    inherits, so a shorthand in one file and a partial override in another
    compose (``bins: [50, 0, 500]`` in the base, ``hi: 800`` in the child).
    It is applied to every file reached through ``_extends`` / ``_include``,
    not to a ``_registry`` pool.

    Pass a set as *sources* to collect every file that was actually read,
    including the ones reached through ``_extends`` / ``_include`` /
    ``_registry``.  :func:`config_sources` uses it so the up-to-date check can
    see a config a set only inherits -- otherwise editing a shared fragment
    leaves every run that includes it silently "up to date".

    Pass a dict as *origins* to learn which files define each name: it maps
    every top-level key (and, for mapping values, ``key/subkey``) to the
    files that set it, base first -- what ``suep-config show`` reports.
    """
    path = Path(path)
    raw = _read(path)
    if sources is not None and path.exists():
        sources.add(path.resolve())
    if not path.exists() and path.name in CONFIG_FILES:
        parent = read_set_file(path.parent).get("extends")
        if parent:
            raw = {"_extends": parent}
            if sources is not None:
                sources.add((path.parent / SET_FILE).resolve())
    _check_directives(raw, path)

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
            base_path, with_fragments=True, sources=sources, normalize=normalize,
            origins=origins, _seen=[*_seen, resolved]))
        axes.update(merged.pop("_axes", {}) or {})

    includes = raw.pop("_include", [])
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        inc_path = Path(inc) if Path(inc).is_absolute() else path.parent / inc
        if not inc_path.exists():
            raise ConfigError(f"ERROR in {path}: _include file not found: {inc_path}")
        merged = _merge(merged, load_config_file(
            inc_path, with_fragments=True, sources=sources, normalize=normalize,
            origins=origins, _seen=[*_seen, resolved]))
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
    if normalize is not None:
        defs = normalize(defs, path, axes)
    if origins is not None:
        for key, value in defs.items():
            if str(key).startswith("_"):
                continue
            origins.setdefault(str(key), []).append(str(path))
            if isinstance(value, dict):
                for sub in value:
                    origins.setdefault(f"{key}/{sub}", []).append(str(path))

    out = _merge(merged, defs)
    if not with_fragments:
        out = {k: v for k, v in out.items() if not str(k).startswith("_")}
    return out


# The files a config set is made of.  variables.yaml is optional: a set's own
# overrides of the global variable catalog (see GLOBAL_VARIABLES).
CONFIG_FILES = ("samples.yaml", "histograms.yaml", "selections.yaml",
                "corrections.yaml", "reweights.yaml", "derived_plots.yaml",
                "columns.yaml", "variables.yaml")

# The variable catalog every set reads: expression, binning and axis title of
# each variable, written once (next to datasets.yaml at the repo root).
GLOBAL_VARIABLES = REPO_ROOT / "variables.yaml"

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
    if "variables.yaml" in files and GLOBAL_VARIABLES.exists():
        sources.add(GLOBAL_VARIABLES.resolve())
    return sources


def fill_sources(config_dir: str | Path) -> list:
    """Every file that decides what a run fills, in a stable order.

    The fill-relevant YAML the set reads (see :func:`config_sources`) and
    every module of ``custom/`` -- the helpers live in sibling files, and
    editing one changes the fills as much as editing ``columns.py``.  The
    up-to-date check compares their mtimes, the resubmission check their
    hashes.
    """
    return sorted(config_sources(config_dir, FILL_CONFIG_FILES)) \
        + sorted((REPO_ROOT / "custom").glob("*.py"))
