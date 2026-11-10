"""coffea processor: reads MDSNano via NanoEvents, fills histograms.

``SuepProcessor`` is a :class:`coffea.processor.ProcessorABC` whose ``process``
method fills the YAML-defined histograms for one chunk of NanoEvents.  It is
driven by :class:`coffea.processor.Runner` (one call per sample, keeping the
per-sample Slurm-array model), which handles file opening, schema application,
chunking and accumulation.  Event weights are managed with
:class:`coffea.analysis_tools.Weights`; corrections are registered via
:mod:`suep_plot.corrections`.  :func:`run_all` is ``suep-run``.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import awkward as ak
import numpy as np

# Applies the scipy shim before coffea.lookup_tools is imported anywhere.
from . import _compat  # noqa: F401

from coffea import processor
from coffea.analysis_tools import Weights
from coffea.nanoevents import NanoAODSchema

from .checks import check_expressions, static_errors
from .config import fill_sources
from .configset import load_config_set
from .corrections import apply_corrections, build_correctors
from .fill import build_histograms, compile_expr, fill_histograms, prepare_events
from .inputs import resolve_sample_files, select_samples
from .reweights import apply_reweights
from .status import output_name


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

        events = prepare_events(events, self.derive_fn, self.settings,
                                self.sel_defs, dataset)

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
        for sname, selection in self.sel_defs.items():
            sel = sel_cache.get(sname)
            if sel is None:
                # Already evaluated once as a selection column, so this cannot
                # fail where the fill did not.
                sel = compile_expr(selection.expression)(events)
                if isinstance(sel, ak.Array):
                    sel = ak.fill_none(sel, False)
            if isinstance(sel, ak.Array) and sel.ndim > 1:
                sel = ak.any(sel, axis=1)
            mask = np.asarray(sel, dtype=bool)
            cutflow[sname] = {"raw": int(mask.sum()), "wtd": float(wvec[mask].sum())}
        return cutflow

    def postprocess(self, accumulator):
        return accumulator


def _inputs_mtime(files: list[str], config_dir: Path,
                  extra_paths: tuple = ()) -> float | None:
    """Newest mtime among configs, custom columns, and local input files.

    Returns ``None`` when freshness can't be established (remote/missing
    inputs), in which case the sample is always processed.
    """
    paths = fill_sources(config_dir) + [Path(p) for p in extra_paths]

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
    cs = load_config_set(config_dir, verbose=True)
    hist_defs, sel_defs, corr_defs = cs.histograms, cs.selections, cs.corrections
    reweighters, derive_fn, settings = cs.reweighters, cs.derive_fn, cs.settings
    if reweighters:
        print(f"Loaded {len(reweighters)} reweight(s): {list(reweighters)}")

    errors = static_errors(cs)
    if errors:
        raise SystemExit("ERROR in " + str(config_dir) + ":\n"
                         + "\n".join(f"  {e}" for e in errors))
    for hname, h in hist_defs.items():
        wspec = h.weight or ""
        if wspec.startswith("@") and not reweighters[wspec[1:]].fill_only:
            print(f"WARNING: histogram '{hname}' uses '{wspec}' which is also applied "
                  f"globally; add 'fill_only: true' to the map to avoid double counting")

    sample_defs = select_samples(cs.samples, samples_filter)

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
    """The definitions in flat form, with ``per_object`` as the fill found it.

    This is what the pickle stores (see ``Histogram.to_dict``), so tools that
    read the histograms back (``suep-plot``, ``suep-reweight``) need neither
    the config set nor to re-evaluate anything.  A histogram never filled has
    no ``per_object``.
    """
    per_object = {n: True for n in layout.get("object", ())}
    per_object.update({n: False for n in layout.get("event", ())})
    return {name: ({**h.to_dict(), "per_object": per_object[name]}
                   if name in per_object else h.to_dict())
            for name, h in hist_defs.items()}
