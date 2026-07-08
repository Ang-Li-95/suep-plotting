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

import importlib
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

from .corrections import apply_corrections, build_correctors, load_correction_defs
from .histograms import (
    _compile_expr,
    build_histograms,
    fill_histograms,
    load_histogram_defs,
    load_selection_defs,
)
from .reweights import apply_reweights, build_reweighters, load_reweight_defs

# MDSNano is a custom NanoAOD flavour; suppress cross-reference warnings for
# collections without the standard NanoAOD index branches.
NanoAODSchema.warn_missing_crossrefs = False


def load_samples(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_files(file_specs: list[str]) -> list[str]:
    """Expand globs / pass through xrootd URLs -> flat list of file paths."""
    import glob

    out = []
    for spec in file_specs:
        if "://" in spec:
            out.append(spec)
        elif "*" in spec or "?" in spec:
            out.extend(sorted(glob.glob(spec)))
        else:
            out.append(spec)
    return out


def _load_custom_columns():
    """Import custom/columns.py if present -> (derive_fn or None)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / "custom" / "columns.py").exists():
        return None
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module("custom.columns")
        importlib.reload(mod)
        return getattr(mod, "derive", None)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load custom/columns.py: {e}")
        return None


class SuepProcessor(processor.ProcessorABC):
    """Fill YAML-defined histograms for one sample chunk."""

    def __init__(self, hist_defs, sel_defs, correctors, sample_defs, derive_fn=None,
                 reweighters=None):
        self.hist_defs = hist_defs
        self.sel_defs = sel_defs
        self.correctors = correctors
        self.sample_defs = sample_defs
        self.derive_fn = derive_fn
        self.reweighters = reweighters or {}

    def process(self, events):
        dataset = events.metadata["dataset"]
        cfg = self.sample_defs.get(dataset, {})
        is_data = cfg.get("is_data", False)
        group = cfg.get("group", "")

        if self.derive_fn is not None:
            events = self.derive_fn(events)

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
        sel_cache = fill_histograms(histograms, self.hist_defs, self.sel_defs,
                                    events, dataset, wvec,
                                    reweighters=self.reweighters)

        cutflow = self._cutflow(events, sel_cache, wvec, n)

        sumw = 0.0 if is_data else float(ak.sum(events.genWeight))
        return {
            "histograms": histograms,
            "sumw": {dataset: sumw},
            "nevents": {dataset: n},
            "cutflow": {dataset: cutflow},
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
                try:
                    sel = _compile_expr(scfg["expression"])(events)
                except Exception:  # noqa: BLE001 - typos already warned at validation
                    continue
                if isinstance(sel, ak.Array):
                    sel = ak.fill_none(sel, False)
            if isinstance(sel, ak.Array) and sel.ndim > 1:
                sel = ak.any(sel, axis=1)
            mask = np.asarray(sel, dtype=bool)
            cutflow[sname] = {"raw": int(mask.sum()), "wtd": float(wvec[mask].sum())}
        return cutflow

    def postprocess(self, accumulator):
        return accumulator


def _validate_expressions(files, tree, hist_defs, sel_defs, corr_defs, derive_fn=None,
                          reweighters=None):
    """Best-effort check: evaluate each expression on a small slice and warn.

    Restores clear feedback for typo'd fields, which otherwise silently produce
    empty histograms.
    """
    from coffea.nanoevents import NanoEventsFactory

    from .histograms import _compile_expr

    if not files:
        return
    # "virtual" mode reads branches lazily, so validation only touches the
    # branches the expressions actually use (an eager open materializes every
    # branch — seconds of wasted I/O on files with PFCand etc.).
    try:
        events = NanoEventsFactory.from_root(
            {files[0]: tree}, mode="virtual", schemaclass=NanoAODSchema,
            entry_stop=200,
        ).events()
    except TypeError:
        try:
            events = NanoEventsFactory.from_root(
                {files[0]: tree}, mode="eager", schemaclass=NanoAODSchema,
                entry_stop=200,
            ).events()
        except Exception as e:  # noqa: BLE001
            print(f"  (skipped expression validation: {e})")
            return
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped expression validation: {e})")
        return

    if derive_fn is not None:
        try:
            events = derive_fn(events)
        except Exception as e:  # noqa: BLE001
            print(f"  (custom derive() failed during validation: {e})")

    bad = []
    for section, defs in (("histogram", hist_defs), ("selection", sel_defs)):
        for name, cfg in defs.items():
            for key in ("expression", "expression_x", "expression_y", "weight"):
                expr = cfg.get(key)
                if not expr or expr.startswith("@"):
                    continue
                try:
                    _compile_expr(expr)(events)
                except Exception as e:  # noqa: BLE001
                    bad.append(f"    {section} '{name}'.{key}: {type(e).__name__}: {str(e)[:80]}")
    for name, cfg in corr_defs.items():
        for expr in cfg.get("inputs", []):
            try:
                _compile_expr(expr)(events)
            except Exception as e:  # noqa: BLE001
                bad.append(f"    correction '{name}' input '{expr}': {type(e).__name__}: {str(e)[:60]}")
    for name, rw in (reweighters or {}).items():
        for expr in rw.exprs:
            try:
                _compile_expr(expr)(events)
            except Exception as e:  # noqa: BLE001
                bad.append(f"    reweight '{name}' expression '{expr}': {type(e).__name__}: {str(e)[:60]}")
    if bad:
        print("WARNING: some config expressions failed to evaluate:")
        print("\n".join(bad))


def _inputs_mtime(files: list[str], config_dir: Path,
                  extra_paths: tuple = ()) -> float | None:
    """Newest mtime among configs, custom columns, and local input files.

    Returns ``None`` when freshness can't be established (remote/missing
    inputs), in which case the sample is always processed.
    """
    import os

    paths = [config_dir / f for f in
             ("samples.yaml", "histograms.yaml", "selections.yaml",
              "corrections.yaml", "reweights.yaml")]
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths.append(repo_root / "custom" / "columns.py")
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
):
    """Load configs, process each sample through coffea, save per-sample pickles.

    *file_range* restricts processing to a ``[start, end)`` slice of each
    sample's resolved file list and *part* tags the output pickle as
    ``<sample>.part<part>.pkl`` — together they let Slurm array tasks split a
    sample by files (one shard per task).  ``suep-plot`` sums the part pickles
    back into one sample at load time.
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

    unknown_maps = sorted(
        (hname, hcfg["weight"])
        for hname, hcfg in hist_defs.items()
        if str(hcfg.get("weight", "")).startswith("@")
        and hcfg["weight"][1:] not in reweighters
    )
    if unknown_maps:
        lines = "\n".join(f"  histogram '{h}': unknown reweight map '{m}'"
                          for h, m in unknown_maps)
        raise SystemExit(
            f"ERROR: histograms.yaml references reweight maps not defined in reweights.yaml:\n{lines}"
        )
    for hname, hcfg in hist_defs.items():
        wspec = str(hcfg.get("weight", ""))
        if wspec.startswith("@") and not reweighters[wspec[1:]].fill_only:
            print(f"WARNING: histogram '{hname}' uses '{wspec}' which is also applied "
                  f"globally; add 'fill_only: true' to the map to avoid double counting")

    unknown_sels = sorted(
        (hname, sname)
        for hname, hcfg in hist_defs.items()
        for sname in hcfg.get("selections", [])
        if sname not in sel_defs
    )
    if unknown_sels:
        lines = "\n".join(f"  histogram '{h}': unknown selection '{s}'"
                          for h, s in unknown_sels)
        raise SystemExit(
            f"ERROR: histograms.yaml references selections not defined in selections.yaml:\n{lines}"
        )

    if samples_filter:
        unknown = [s for s in samples_filter if s not in sample_defs]
        if unknown:
            print(f"WARNING: unknown sample(s) ignored: {unknown}")
        sample_defs = {k: v for k, v in sample_defs.items() if k in samples_filter}

    if not sample_defs:
        print("No samples to process.")
        return output_dir

    correctors = {}
    if corr_defs:
        try:
            correctors = build_correctors(corr_defs)
            print(f"Loaded {len(correctors)} correction(s): {list(correctors)}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not load corrections: {e}")
            print("         proceeding without corrections")

    derive_fn = _load_custom_columns()
    if derive_fn is not None:
        print("Custom columns: loaded derive()")

    proc = SuepProcessor(hist_defs, sel_defs, correctors, sample_defs, derive_fn,
                         reweighters)
    rw_files = tuple(str(rw.include_path) for rw in reweighters.values()
                     if rw.include_path is not None)
    if workers and workers > 1:
        executor = processor.FuturesExecutor(workers=workers)
    else:
        executor = processor.IterativeExecutor()
    runner = processor.Runner(executor=executor, schema=NanoAODSchema, chunksize=chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(sample_defs)} sample(s), {len(hist_defs)} histogram(s)")

    validated = False
    for name, cfg in sample_defs.items():
        file_specs = cfg.get("files") or []
        if not file_specs:
            print(f"  WARNING: sample '{name}' has no 'files' entry, skipping")
            continue
        files = _resolve_files(file_specs)
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

        suffix = f".part{part}" if part is not None else ""
        out_path = output_dir / f"{name}{suffix}.pkl"
        if not force and out_path.exists():
            newest = _inputs_mtime(files, config_dir, rw_files)
            if newest is not None and out_path.stat().st_mtime >= newest:
                print(f"  {name}: up to date, skipping (--force to reprocess)")
                continue

        if not validated:
            _validate_expressions(files, tree, hist_defs, sel_defs, corr_defs,
                                  derive_fn, reweighters)
            validated = True

        t0 = time.time()
        out = runner({name: files}, proc, treename=tree)
        elapsed = time.time() - t0
        nev = out.get("nevents", {}).get(name, 0)
        rate = nev / elapsed if elapsed > 0 else 0
        print(f"  {name}: {nev:,} events in {elapsed:.1f}s ({rate:.0f} evt/s)")

        payload = {
            "histograms": out["histograms"],
            "samples": {name: cfg},
            "hist_defs": hist_defs,
            "sumw": out.get("sumw", {}),
            "nevents": out.get("nevents", {}),
            "cutflow": out.get("cutflow", {}),
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    -> {out_path}")

    print(f"Saved sample file(s) to {output_dir}/")
    return output_dir
