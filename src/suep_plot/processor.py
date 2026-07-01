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
    build_histograms,
    fill_histograms,
    load_histogram_defs,
    load_selection_defs,
)

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

    def __init__(self, hist_defs, sel_defs, correctors, sample_defs, derive_fn=None):
        self.hist_defs = hist_defs
        self.sel_defs = sel_defs
        self.correctors = correctors
        self.sample_defs = sample_defs
        self.derive_fn = derive_fn

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

        histograms = build_histograms(self.hist_defs, [dataset])
        fill_histograms(histograms, self.hist_defs, self.sel_defs,
                        events, dataset, weights.weight())

        sumw = 0.0 if is_data else float(ak.sum(events.genWeight))
        return {
            "histograms": histograms,
            "sumw": {dataset: sumw},
            "nevents": {dataset: n},
        }

    def postprocess(self, accumulator):
        return accumulator


def _validate_expressions(files, tree, hist_defs, sel_defs, corr_defs, derive_fn=None):
    """Best-effort check: evaluate each expression on a small slice and warn.

    Restores clear feedback for typo'd fields, which otherwise silently produce
    empty histograms.
    """
    from coffea.nanoevents import NanoEventsFactory

    from .histograms import _compile_expr

    if not files:
        return
    try:
        events = NanoEventsFactory.from_root(
            {files[0]: tree}, mode="eager", schemaclass=NanoAODSchema,
            entry_stop=200,
        ).events()
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
                if not expr:
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
    if bad:
        print("WARNING: some config expressions failed to evaluate:")
        print("\n".join(bad))


def run_all(
    config_dir: str,
    output_dir: str,
    samples_filter: list[str] | None = None,
    chunk_size: int = 100_000,
    workers: int = 1,
):
    """Load configs, process each sample through coffea, save per-sample pickles."""
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    sample_defs = load_samples(config_dir / "samples.yaml")
    hist_defs = load_histogram_defs(config_dir / "histograms.yaml")
    sel_defs = load_selection_defs(config_dir / "selections.yaml")
    corr_defs = load_correction_defs(config_dir / "corrections.yaml")

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

    proc = SuepProcessor(hist_defs, sel_defs, correctors, sample_defs, derive_fn)
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

        if not validated:
            _validate_expressions(files, tree, hist_defs, sel_defs, corr_defs, derive_fn)
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
        }
        out_path = output_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"    -> {out_path}")

    print(f"Saved sample file(s) to {output_dir}/")
    return output_dir
