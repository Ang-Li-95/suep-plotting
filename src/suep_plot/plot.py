"""``suep-plot``: every histogram of a run, drawn and indexed.

:func:`plot_all` merges the run's pickles (:mod:`suep_plot.results`), draws
each histogram (:mod:`suep_plot.figures`) and each derived plot
(:mod:`suep_plot.derived`), in parallel if asked, and writes the gallery
(:mod:`suep_plot.gallery`).
"""

from __future__ import annotations

import os
import time

from .derived import (derived_sources, load_derived_plot_defs, plot_derived,
                      require_sources)
from .figures import DEFAULT_FORMATS, is_2d_hist, plot_histogram, plot_histogram_2d
from .gallery import write_gallery
from .results import (apply_xs_scaling, export_root, merge_results, resolve_inputs,
                      restyle, write_cutflow)


def _render_task(args) -> tuple[str, str | None]:
    """Render one plotting task; returns (name, error-or-None)."""
    task, common = args
    kind, name = task[0], task[1]
    try:
        if kind == "1d":
            _, _, h, cfg = task
            plot_histogram(h, cfg, common["sample_defs"], common["output_dir"], name,
                           normalize=common["normalize"], log_y=common["log_y"],
                           lumi=common["lumi"], formats=common["formats"],
                           ratio=common["ratio"])
        elif kind == "2d":
            _, _, h, cfg = task
            plot_histogram_2d(h, cfg, common["sample_defs"], common["output_dir"], name,
                              lumi=common["lumi"], formats=common["formats"])
        else:  # derived
            _, _, cfg, hists = task
            plot_derived(name, cfg, hists, common["sample_defs"], common["output_dir"],
                              log_y=common["log_y"], lumi=common["lumi"],
                              formats=common["formats"],
                              normalize=common["normalize"], ratio=common["ratio"])
        return name, None
    except Exception as e:  # noqa: BLE001 - reported per plot, run continues
        return name, f"{type(e).__name__}: {e}"


def _run_tasks(tasks: list, common: dict, jobs: int) -> list[tuple[str, str | None]]:
    args = [(t, common) for t in tasks]
    if jobs > 1 and len(args) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        try:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(jobs, len(args)), mp_context=ctx) as ex:
                return list(ex.map(_render_task, args))
        except (ValueError, OSError) as e:
            print(f"  (parallel rendering unavailable: {e}; falling back to serial)")
    return [_render_task(a) for a in args]


def plot_all(
    results_path: str | list[str],
    output_dir: str,
    normalize: bool = False,
    log_y: bool = False,
    lumi: float | None = None,
    config_dir: str | None = None,
    jobs: int = 1,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    ratio: bool = True,
    save_root: str | None = None,
):
    """Plot all histograms from per-sample pickle files.

    results_path can be a single .pkl file, a list of .pkl files,
    a directory containing .pkl files, or a mix of files and directories.
    Per-sample histograms are merged before plotting.  When *lumi* is given,
    MC samples are normalized to xs * lumi * 1000 / sumw.  With *jobs* > 1
    figures render in parallel processes.  *save_root* additionally exports
    every histogram to a ROOT file (after any lumi scaling).

    If *config_dir* is given, plot-time keys from its histograms.yaml (label,
    blind, rebin, flow, log_x, log_y, log_z ...) and its samples.yaml (label,
    color, linestyle, marker ...) override the ones stored in the pickles, so
    styling can be iterated without reprocessing.
    """
    started = time.time()
    pkl_files = resolve_inputs(results_path)
    if not pkl_files:
        raise SystemExit(f"ERROR: no .pkl files found in {results_path}")
    print(f"Loading {len(pkl_files)} sample file(s):")
    for f in pkl_files:
        print(f"  {f}")
    data = merge_results(pkl_files)

    histograms = data["histograms"]
    sample_defs = data.get("samples", {})
    hist_defs = data.get("hist_defs", {})

    if config_dir:
        restyle(hist_defs, sample_defs, config_dir)

    if lumi is not None:
        apply_xs_scaling(histograms, sample_defs, data.get("sumw", {}), lumi)

    os.makedirs(output_dir, exist_ok=True)
    write_cutflow(data.get("cutflow", {}), output_dir)
    if save_root:
        export_root(histograms, save_root)

    common = {
        "sample_defs": sample_defs,
        "output_dir": output_dir,
        "normalize": normalize,
        "log_y": log_y,
        "lumi": lumi,
        "formats": tuple(formats),
        "ratio": ratio,
    }

    tasks = []
    for name, h in histograms.items():
        kind = "2d" if is_2d_hist(h) else "1d"
        tasks.append((kind, name, h, hist_defs.get(name, {})))
    n_1d = sum(1 for t in tasks if t[0] == "1d")
    n_2d = sum(1 for t in tasks if t[0] == "2d")

    derived_defs = {}
    if config_dir:
        derived_defs = load_derived_plot_defs(os.path.join(config_dir, "derived_plots.yaml"))
    for name, cfg in derived_defs.items():
        needed = derived_sources(name, cfg)
        if needed is None or not require_sources(histograms, needed, name):
            continue
        tasks.append(("derived", name, cfg, {s: histograms[s] for s in needed}))

    n_derived = len(tasks) - n_1d - n_2d
    mode = f"{jobs} process(es)" if jobs > 1 else "serial"
    print(f"Plotting {n_1d} 1D + {n_2d} 2D + {n_derived} derived plot(s) "
          f"to {output_dir}/ [{mode}]")

    failed = 0
    for name, err in _run_tasks(tasks, common, jobs):
        if err:
            failed += 1
            print(f"  WARNING: '{name}' failed: {err}")
        else:
            print(f"  {name}")

    gallery = write_gallery(output_dir, formats=formats, since=started)
    if gallery:
        print(f"Gallery: {gallery}")

    if failed:
        print(f"Done ({failed} plot(s) failed).")
    else:
        print("Done.")
    return failed
