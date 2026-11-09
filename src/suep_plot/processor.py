"""coffea processor: reads MDSNano via NanoEvents, fills histograms.

``SuepProcessor`` is a :class:`coffea.processor.ProcessorABC` whose ``process``
method fills the YAML-defined histograms for one chunk of NanoEvents.  It is
driven by :class:`coffea.processor.Runner` (one call per sample, keeping the
per-sample Slurm-array model), which handles file opening, schema application,
chunking and accumulation.  Event weights are managed with
:class:`coffea.analysis_tools.Weights`; corrections are registered via
:mod:`suep_plot.corrections`.
"""

from __future__ import annotations

import difflib
import importlib
import os
import pickle
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import yaml

# Applies the scipy shim before coffea.lookup_tools is imported anywhere.
from . import _compat  # noqa: F401

from coffea import processor
from coffea.analysis_tools import Weights
from coffea.nanoevents import NanoAODSchema

from .config import FILL_CONFIG_FILES, config_sources, load_config_file
from .corrections import apply_corrections, build_correctors, load_correction_defs
from .histograms import (
    _compile_expr,
    attach_selection_columns,
    build_histograms,
    fill_histograms,
    load_histogram_defs,
    load_selection_defs,
)
from .reweights import apply_reweights, build_reweighters, load_reweight_defs
from .status import output_name

# MDSNano is a custom NanoAOD flavour; suppress cross-reference warnings for
# collections without the standard NanoAOD index branches.
NanoAODSchema.warn_missing_crossrefs = False


def load_samples(path: str) -> dict:
    """Read a config's ``samples.yaml`` -> {name: cfg}.

    Definitions normally come from the shared registry named by ``_registry:``;
    see :mod:`suep_plot.config` for that and the other config directives.
    """
    return load_config_file(path)


def select_samples(sample_defs: dict, wanted: list[str] | None) -> dict:
    """The samples to process: all of them, or the ones named in *wanted*.

    Names may also be given comma-separated (``-s a,b``).  An unknown name is
    fatal -- silently processing fewer samples than asked for looks like a
    successful run -- and the message lists what the set does define.
    """
    if not sample_defs:
        raise SystemExit("ERROR: samples.yaml defines no samples")
    if not wanted:
        return dict(sample_defs)
    names = [n.strip() for spec in wanted for n in str(spec).split(",") if n.strip()]
    unknown = [n for n in names if n not in sample_defs]
    if unknown:
        hints = []
        for n in unknown:
            close = difflib.get_close_matches(n, list(sample_defs), n=1)
            if close:
                hints.append(f"'{n}' -> did you mean '{close[0]}'?")
        hint = ("\n  " + "\n  ".join(hints)) if hints else ""
        raise SystemExit(f"ERROR: unknown sample(s) {unknown}{hint}\n"
                         f"  available: {', '.join(sample_defs)}")
    return {k: v for k, v in sample_defs.items() if k in names}


def _split_xrootd_url(url: str) -> tuple[str, str]:
    """``root://host//eos/dir`` -> (``root://host/``, ``/eos/dir``)."""
    scheme, rest = url.split("://", 1)
    host, _, path = rest.partition("/")
    return f"{scheme}://{host}/", "/" + path.lstrip("/")


def _xrootd_listdir(fs, path: str) -> list[str]:
    """Recursively list ``*.root`` under a remote directory (server paths)."""
    from XRootD.client.flags import DirListFlags

    status, listing = fs.dirlist(path, DirListFlags.STAT)
    if not status.ok or listing is None:
        raise RuntimeError(f"xrootd dirlist failed for {path}: {status.message}")

    out = []
    for entry in listing:
        child = f"{path.rstrip('/')}/{entry.name}"
        if entry.statinfo is not None and entry.statinfo.flags & 2:  # kXR_isDir
            out.extend(_xrootd_listdir(fs, child))
        elif entry.name.endswith(".root"):
            out.append(child)
    return sorted(out)


def _resolve_xrootd(spec: str) -> list[str]:
    """Expand one xrootd spec: a file passes through, a directory is walked,
    a trailing wildcard is matched against its parent directory's listing."""
    import fnmatch

    from XRootD import client

    prefix, path = _split_xrootd_url(spec)
    if "*" not in path and "?" not in path and path.endswith(".root"):
        return [spec]

    fs = client.FileSystem(prefix)
    # Server paths are absolute, so host + path keeps the usual root://host//eos/…
    if "*" in path or "?" in path:
        parent, _, pattern = path.rpartition("/")
        return [prefix + f for f in _xrootd_listdir(fs, parent)
                if fnmatch.fnmatch(f.rpartition("/")[2], pattern)]
    return [prefix + f for f in _xrootd_listdir(fs, path)]


def _resolve_files(file_specs: list[str]) -> list[str]:
    """Expand globs / directories / xrootd URLs -> flat list of file paths.

    A spec may be a single file, a glob, or a directory (local or xrootd), in
    which case it is walked recursively for ``*.root``.
    """
    import glob
    import os

    out = []
    for spec in file_specs:
        if "://" in spec:
            out.extend(_resolve_xrootd(spec))
        elif "*" in spec or "?" in spec:
            out.extend(sorted(glob.glob(spec)))
        elif os.path.isdir(spec):
            out.extend(sorted(str(p) for p in Path(spec).rglob("*.root")))
        else:
            out.append(spec)
    return out


def resolve_sample_files(cfg: dict) -> list[str]:
    """The files of one sample, honouring an optional ``max_files:`` cap.

    A background sample can be far larger than the comparison needs -- DY is
    5,575 files against a signal's 25 -- and a normalized shape overlay gains
    nothing from the tail of it.  ``max_files`` takes the first N of the
    resolved list, deterministically (the list is sorted), so a capped run is
    reproducible and can be widened later without redoing what exists.
    """
    files = _resolve_files(cfg.get("files") or [])
    cap = cfg.get("max_files")
    return files[:int(cap)] if cap else files


def load_columns_config(path: str) -> dict:
    """Read a config's optional ``columns.yaml`` -> settings for ``derive()``.

    Missing file = ``{}`` = the module's own defaults.  The schema is the
    custom-columns module's business (``parameters`` / ``steps`` for the MDS
    module); it is only forwarded here.
    """
    return load_config_file(path)


def _import_custom():
    """Import the ``custom.columns`` module that ships next to the package.

    Fatal when it is missing or does not import: every config set's derived
    collections (the clusters, the LLPs) come from it, so running without it
    would only produce empty histograms.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / "custom" / "columns.py").exists():
        raise SystemExit(
            f"ERROR: custom/columns.py not found next to the suep_plot package "
            f"(looked in {repo_root}).\n  suep-plotting must be installed "
            "editable from its checkout:  pip install -e <checkout>")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module("custom.columns")
        # columns.py imports its helpers from sibling modules, so reload the
        # whole package: reloading columns.py alone would re-bind its names to
        # the already-cached submodules and an edit there would be ignored.
        # (This is for a live notebook kernel; a fresh suep-run imports once.)
        # In dependency order -- params.py owns the schema the other modules
        # import, so it goes first and columns.py last.
        helpers = sorted(n for n in sys.modules if n.startswith("custom.")
                         and n not in ("custom.columns", "custom.params"))
        for name in ["custom.params", *helpers]:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        importlib.reload(mod)
    except Exception as e:  # noqa: BLE001 - re-raised as a fatal, explained error
        raise SystemExit(
            f"ERROR: custom/columns.py failed to import: {type(e).__name__}: {e}\n"
            "  It needs scikit-learn, numba and awkward in the active "
            "environment (conda activate rootmds).") from e
    return mod


def _load_custom_columns(columns_cfg: dict | None = None, verbose: bool = True):
    """Import custom/columns.py -> (derive_fn, settings).

    The settings are resolved here, once, and handed to the processor -- which
    passes them to ``derive()`` and pickles them to its workers, so a worker
    cannot end up running on settings its parent never chose.  A bad
    columns.yaml is fatal: it changes what the derived columns *mean*, so it
    must not degrade to a warning and silently different histograms.
    """
    mod = _import_custom()
    derive_fn = getattr(mod, "derive", None)
    configure = getattr(mod, "configure", None)
    if configure is None:
        if columns_cfg:
            raise SystemExit("ERROR: columns.yaml given, but custom/columns.py "
                             "has no configure() to read it")
        return derive_fn, None
    try:
        settings = configure(columns_cfg)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ERROR in columns.yaml: {e}")
    if derive_fn is not None and verbose:
        print("Custom columns: loaded derive()")
        print(f"                steps  {list(settings.steps)}")
        print(f"                params {settings.params}")
    return derive_fn, settings


def _custom_config_errors(settings, sel_defs: dict) -> list[str]:
    """What custom/columns.py's ``validate()`` finds wrong with the set, if anything.

    Problems ``derive()`` would otherwise hit only mid-run -- e.g. the
    ``cluster_isolation`` step without the prompt-object selections it reads.
    """
    validate = getattr(_import_custom(), "validate", None)
    if validate is None or settings is None:
        return []
    return list(validate(settings, sel_defs))


def _reference_errors(hist_defs: dict, sel_defs: dict, reweighters: dict) -> list[str]:
    """Names a histogram uses that the set does not define."""
    errors = []
    for hname, hcfg in sorted(hist_defs.items()):
        wspec = str(hcfg.get("weight", ""))
        if wspec.startswith("@") and wspec[1:] not in reweighters:
            errors.append(f"histogram '{hname}': unknown reweight map '{wspec}' "
                          "(not in reweights.yaml)")
        for sname in hcfg.get("selections", []):
            if sname not in sel_defs:
                close = difflib.get_close_matches(sname, list(sel_defs), n=1)
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                errors.append(f"histogram '{hname}': unknown selection "
                              f"'{sname}'{hint}")
    return errors


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
    errors += _reference_errors(hist_defs, sel_defs, reweighters)
    try:
        _, settings = _load_custom_columns(
            load_columns_config(config_dir / "columns.yaml"), verbose=False)
        errors += _custom_config_errors(settings, sel_defs)
    except SystemExit as e:
        errors.append(str(e))
    return errors


def load_config_set(config_dir: str | Path) -> dict:
    """Everything a run reads from one config set, loaded and resolved.

    Keys: ``samples``, ``histograms``, ``selections``, ``corrections``,
    ``reweighters``, ``derive_fn``, ``settings``.
    """
    config_dir = Path(config_dir)
    try:
        reweighters = build_reweighters(
            load_reweight_defs(config_dir / "reweights.yaml"), config_dir)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"ERROR in reweights.yaml: {e}")
    derive_fn, settings = _load_custom_columns(
        load_columns_config(config_dir / "columns.yaml"), verbose=False)
    return {
        "samples": load_samples(config_dir / "samples.yaml"),
        "histograms": load_histogram_defs(config_dir / "histograms.yaml"),
        "selections": load_selection_defs(config_dir / "selections.yaml"),
        "corrections": load_correction_defs(config_dir / "corrections.yaml"),
        "reweighters": reweighters,
        "derive_fn": derive_fn,
        "settings": settings,
    }


def config_warnings(config_dir: str | Path) -> list[str]:
    """Things that work but are probably not what was meant."""
    from .config import CONFIG_FILES, _read, read_set_file
    from .plot import derived_sources, load_derived_plot_defs

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


def _write_pickle_atomic(path: Path, payload) -> None:
    """Write *payload* so that *path* is either the old file or the whole new one.

    A task killed mid-write leaves only the temporary file (which ends in
    ``.tmp.<pid>``, so no ``*.pkl`` glob picks it up), never a truncated pickle
    under the real name.
    """
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


class SuepProcessor(processor.ProcessorABC):
    """Fill YAML-defined histograms for one sample chunk."""

    def __init__(self, hist_defs, sel_defs, correctors, sample_defs, derive_fn=None,
                 reweighters=None, settings=None):
        self.hist_defs = hist_defs
        self.sel_defs = sel_defs
        self.correctors = correctors
        self.sample_defs = sample_defs
        self.derive_fn = derive_fn
        self.reweighters = reweighters or {}
        # The resolved columns.yaml.  Pickled to the workers along with the
        # rest of the processor, so a worker cannot run on other settings.
        self.settings = settings

    def process(self, events):
        dataset = events.metadata["dataset"]
        cfg = self.sample_defs.get(dataset, {})
        is_data = cfg.get("is_data", False)
        group = cfg.get("group", "")

        if self.derive_fn is not None:
            events = self.derive_fn(events, self.settings, self.sel_defs)
        # After derive(), so a selection over a derived collection becomes a
        # column too.  See histograms.attach_selection_columns.
        events = attach_selection_columns(events, self.sel_defs, dataset)

        n = len(events)
        weights = Weights(n, storeIndividual=False)
        if is_data:
            weights.add("genWeight", np.ones(n, dtype=np.float64))
        else:
            weights.add("genWeight", np.asarray(events.genWeight, dtype=np.float64))
            if self.correctors:
                apply_corrections(self.correctors, events, group, weights)
        if self.reweighters:
            apply_reweights(self.reweighters, events, dataset, group, is_data, weights)

        histograms = build_histograms(self.hist_defs, [dataset])
        wvec = weights.weight()
        layout: dict = {}
        sel_cache = fill_histograms(histograms, self.hist_defs, self.sel_defs,
                                    events, dataset, wvec,
                                    reweighters=self.reweighters, layout=layout)

        cutflow = self._cutflow(events, sel_cache, wvec, n)

        sumw = 0.0 if is_data else float(ak.sum(events.genWeight))
        return {
            "histograms": histograms,
            "sumw": {dataset: sumw},
            "nevents": {dataset: n},
            "cutflow": {dataset: cutflow},
            # Sets, because coffea accumulates them by union across chunks.
            "layout": {"object": {k for k, v in layout.items() if v},
                       "event": {k for k, v in layout.items() if not v}},
        }

    def _cutflow(self, events, sel_cache, wvec, n):
        """Raw and weighted event counts passing each named selection.

        Counts are independent per selection (not sequential).  Object-level
        selections count events with >= 1 passing object.
        """
        cutflow = {"total": {"raw": int(n), "wtd": float(wvec.sum())}}
        for sname, scfg in self.sel_defs.items():
            sel = sel_cache.get(sname)
            if sel is None:
                # Already evaluated once as a selection column, so this cannot
                # fail where the fill did not.
                sel = _compile_expr(scfg["expression"])(events)
                if isinstance(sel, ak.Array):
                    sel = ak.fill_none(sel, False)
            if isinstance(sel, ak.Array) and sel.ndim > 1:
                sel = ak.any(sel, axis=1)
            mask = np.asarray(sel, dtype=bool)
            cutflow[sname] = {"raw": int(mask.sum()), "wtd": float(wvec[mask].sum())}
        return cutflow

    def postprocess(self, accumulator):
        return accumulator


def open_events(path: str, tree: str = "Events", n: int = 200):
    """The first *n* events of one file as NanoEvents, read lazily."""
    from coffea.nanoevents import NanoEventsFactory

    # "virtual" mode reads branches lazily, so a check only touches the
    # branches the expressions actually use (an eager open materializes every
    # branch -- seconds of wasted I/O on files with PFCand etc.).
    try:
        return NanoEventsFactory.from_root(
            {path: tree}, mode="virtual", schemaclass=NanoAODSchema,
            entry_stop=n).events()
    except TypeError:
        return NanoEventsFactory.from_root(
            {path: tree}, mode="eager", schemaclass=NanoAODSchema,
            entry_stop=n).events()


def prepare_events(events, derive_fn, settings, sel_defs, sample: str = "?"):
    """What every expression sees: derived columns, then selection columns."""
    if derive_fn is not None:
        events = derive_fn(events, settings, sel_defs)
    return attach_selection_columns(events, sel_defs, sample)


def check_expressions(files, tree, hist_defs, sel_defs, corr_defs, derive_fn=None,
                      reweighters=None, settings=None, sample: str = "?") -> list[str]:
    """Evaluate every expression on the first events of *files*; return the failures.

    Each histogram's and selection's expression must evaluate, and an explicit
    ``per_object`` / ``level`` must agree with what the expression gives, on
    every sample -- the fill treats any failure as fatal, so finding them here
    stops the run in seconds instead of on some chunk an hour in.
    """
    from .histograms import is_2d, is_jagged

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

    for name, cfg in sel_defs.items():
        mask = evaluate(f"selection '{name}'", cfg["expression"])
        level = cfg.get("level")
        if mask is not None and level is not None \
                and (level == "object") != is_jagged(mask):
            bad.append(f"selection '{name}': declared 'level: {level}' but the "
                       "mask says otherwise (drop the key, it is inferred)")
    for name, cfg in hist_defs.items():
        keys = ("expression_x", "expression_y") if is_2d(cfg) else ("expression",)
        values = [evaluate(f"histogram '{name}' {k}", cfg[k]) for k in keys]
        if all(v is not None for v in values):
            jagged = {is_jagged(v) for v in values}
            declared = cfg.get("per_object")
            if len(jagged) > 1:
                bad.append(f"histogram '{name}': one axis per object, the other "
                           "per event")
            elif declared is not None and bool(declared) != jagged.pop():
                bad.append(f"histogram '{name}': declared 'per_object: {declared}' "
                           "but the expression says otherwise (drop the key)")
        weight = cfg.get("weight")
        if weight and not str(weight).startswith("@"):
            evaluate(f"histogram '{name}' weight", weight)
    for name, cfg in corr_defs.items():
        for expr in cfg.get("inputs", []):
            evaluate(f"correction '{name}' input", expr)
    for name, rw in (reweighters or {}).items():
        for expr in rw.exprs:
            evaluate(f"reweight '{name}'", expr)
    return bad


def _inputs_mtime(files: list[str], config_dir: Path,
                  extra_paths: tuple = ()) -> float | None:
    """Newest mtime among configs, custom columns, and local input files.

    Returns ``None`` when freshness can't be established (remote/missing
    inputs), in which case the sample is always processed.
    """
    import os

    # Every file the set actually reads, not just the ones in its directory:
    # a set that _extends a reference set or _includes a shared fragment is
    # out of date when *those* change too (see config.config_sources).
    paths = list(config_sources(config_dir, FILL_CONFIG_FILES))
    repo_root = Path(__file__).resolve().parent.parent.parent
    # Every module of the custom-columns package, not just columns.py: the
    # helpers live in sibling files and editing one changes the fills too.
    paths.extend(sorted((repo_root / "custom").glob("*.py")))
    paths.extend(Path(p) for p in extra_paths)

    newest = 0.0
    for p in paths:
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    for f in files:
        if "://" in f:
            return None
        try:
            newest = max(newest, os.path.getmtime(f))
        except OSError:
            return None
    return newest


def run_all(
    config_dir: str,
    output_dir: str,
    samples_filter: list[str] | None = None,
    chunk_size: int = 100_000,
    workers: int = 1,
    force: bool = False,
    file_range: tuple[int, int] | None = None,
    part: str | int | None = None,
    file_list: list[str] | None = None,
):
    """Load configs, process each sample through coffea, save per-sample pickles.

    *file_list* replaces the sample's own ``files:`` with an already-resolved
    list of files, and *part* tags the output pickle as
    ``<sample>.part<part>.pkl``; ``suep-submit`` uses the pair to hand each
    array task a frozen shard of one sample (``suep-plot`` sums the part
    pickles back into one sample at load time).  *file_range* is the
    equivalent for a direct ``suep-run``: a ``[start, end)`` slice of the
    resolved file list.
    """
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    sample_defs = load_samples(config_dir / "samples.yaml")
    hist_defs = load_histogram_defs(config_dir / "histograms.yaml")
    sel_defs = load_selection_defs(config_dir / "selections.yaml")
    corr_defs = load_correction_defs(config_dir / "corrections.yaml")
    rw_defs = load_reweight_defs(config_dir / "reweights.yaml")

    # Reweighting changes physics results: config errors are fatal, not warnings.
    try:
        reweighters = build_reweighters(rw_defs, config_dir)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"ERROR in reweights.yaml: {e}")
    if reweighters:
        print(f"Loaded {len(reweighters)} reweight(s): {list(reweighters)}")

    errors = _reference_errors(hist_defs, sel_defs, reweighters)
    if errors:
        raise SystemExit("ERROR in " + str(config_dir) + ":\n"
                         + "\n".join(f"  {e}" for e in errors))
    for hname, hcfg in hist_defs.items():
        wspec = str(hcfg.get("weight", ""))
        if wspec.startswith("@") and not reweighters[wspec[1:]].fill_only:
            print(f"WARNING: histogram '{hname}' uses '{wspec}' which is also applied "
                  f"globally; add 'fill_only: true' to the map to avoid double counting")

    sample_defs = select_samples(sample_defs, samples_filter)

    correctors = {}
    if corr_defs:
        # Fatal, like a bad reweight: carrying on without the scale factors
        # would fill histograms that only look right.
        try:
            correctors = build_correctors(corr_defs)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"ERROR: could not load corrections.yaml: "
                             f"{type(e).__name__}: {e}")
        print(f"Loaded {len(correctors)} correction(s): {list(correctors)}")

    columns_cfg = load_columns_config(config_dir / "columns.yaml")
    derive_fn, settings = _load_custom_columns(columns_cfg)
    errors = _custom_config_errors(settings, sel_defs)
    if errors:
        raise SystemExit("ERROR in " + str(config_dir) + ":\n"
                         + "\n".join(f"  {e}" for e in errors))

    remote = (file_list or []) + [str(s) for cfg in sample_defs.values()
                                  for s in (cfg.get("files") or [])]
    if any("://" in f for f in remote):
        from .slurm import ensure_proxy
        ensure_proxy(need_seconds=60)

    proc = SuepProcessor(hist_defs, sel_defs, correctors, sample_defs, derive_fn,
                         reweighters, settings)
    rw_files = tuple(str(rw.include_path) for rw in reweighters.values()
                     if rw.include_path is not None)
    if workers and workers > 1:
        executor = processor.FuturesExecutor(workers=workers)
    else:
        executor = processor.IterativeExecutor()
    runner = processor.Runner(executor=executor, schema=NanoAODSchema, chunksize=chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(sample_defs)} sample(s), {len(hist_defs)} histogram(s)")

    todo = []
    for name, cfg in sample_defs.items():
        # A caller-supplied list is already resolved: no directory walk here,
        # so a shard processes exactly the files it was submitted with.
        if file_list is not None:
            files = list(file_list)
        else:
            file_specs = cfg.get("files") or []
            if not file_specs:
                print(f"  WARNING: sample '{name}' has no 'files' entry, skipping")
                continue
            files = resolve_sample_files(cfg)
        tree = cfg.get("tree", "Events")
        if not files:
            print(f"  WARNING: no files resolved for '{name}', skipping")
            continue
        if file_range is not None:
            start, end = file_range
            files = files[start:end]
            if not files:
                print(f"  WARNING: file range {start}:{end} selects no files "
                      f"for '{name}', skipping")
                continue

        out_path = output_dir / output_name(name, part)
        if not force and out_path.exists():
            newest = _inputs_mtime(files, config_dir, rw_files)
            if newest is not None and out_path.stat().st_mtime >= newest:
                print(f"  {name}: up to date, skipping (--force to reprocess)")
                continue
        todo.append((name, cfg, files, tree, out_path))

    # Every expression on the first file of every sample, before any of them
    # is processed: a failure is fatal in the fill, so better all at once now.
    problems = []
    for name, cfg, files, tree, _ in todo:
        problems += [f"  [{name}] {p}" for p in check_expressions(
            files, tree, hist_defs, sel_defs, corr_defs, derive_fn, reweighters,
            settings, sample=name)]
    if problems:
        raise SystemExit("ERROR: config expressions that do not work (checked on "
                         "the first events of each sample's first file):\n"
                         + "\n".join(problems))

    for name, cfg, files, tree, out_path in todo:
        t0 = time.time()
        out = runner({name: files}, proc, treename=tree)
        elapsed = time.time() - t0
        nev = out.get("nevents", {}).get(name, 0)
        rate = nev / elapsed if elapsed > 0 else 0
        print(f"  {name}: {nev:,} events in {elapsed:.1f}s ({rate:.0f} evt/s)")

        payload = {
            "histograms": out["histograms"],
            "samples": {name: cfg},
            "hist_defs": _with_layout(hist_defs, out.get("layout", {})),
            "sumw": out.get("sumw", {}),
            "nevents": out.get("nevents", {}),
            "cutflow": out.get("cutflow", {}),
        }
        _write_pickle_atomic(out_path, payload)
        print(f"    -> {out_path}")

    print(f"Saved sample file(s) to {output_dir}/")
    return output_dir


def _with_layout(hist_defs: dict, layout: dict) -> dict:
    """The definitions with ``per_object`` as the fill found it.

    Stored in the pickle, so tools that read the histograms back
    (``suep-reweight``) know which ones are per object without re-evaluating
    anything.  A histogram never filled keeps whatever its config said.
    """
    per_object = {n: True for n in layout.get("object", ())}
    per_object.update({n: False for n in layout.get("event", ())})
    return {name: ({**cfg, "per_object": per_object[name]} if name in per_object
                   else cfg)
            for name, cfg in hist_defs.items()}
