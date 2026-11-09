"""Shared suep-plot styling for the standalone comparison scripts.

The comparison scripts read the same pickles ``suep-plot`` reads, so they reuse
its helpers directly -- through the public names at the end of
``suep_plot.plot`` -- for the mplhep CMS style, the plot-time transforms
declared in ``histograms.yaml`` (rebin / flow folding / log axes), the
per-sample labels and colors from ``samples.yaml``, png+pdf output (into
subfolders for region histograms) and the HTML thumbnail gallery.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mplhep as hep  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suep_plot import plot as _plot  # noqa: E402
from suep_plot.plot import (  # noqa: E402
    DEFAULT_FORMATS,
    merge_results,
    palette,
    save_figure,
)

hep.style.use("CMS")

# Figure geometry comes from suep_plot.plot itself -- these scripts are meant to
# sit next to suep-plot's own output in the same talk, so a divergence here is a
# bug rather than a style choice.  figure(), or figure(ratio=True) for a main
# panel with a ratio panel under it.
figure = _plot.figure
LEGEND_FONTSIZE = 18


def cms_label(ax, lumi=None, has_data=False):
    """CMS 'Preliminary' header at 13.6 TeV (simulation-only by default).

    Pass ``has_data`` when one of the overlaid samples is real data, so the
    header drops "Simulation".
    """
    _plot.cms_label(ax, lumi=lumi, has_data=has_data)


# ── loading ───────────────────────────────────────────────────────


def _pickles(outdir: str) -> list[str]:
    """The pickles suep-plot would merge from *outdir* (a run's own tasks only)."""
    return _plot.resolve_inputs(outdir)


def samples_in(outdir: str) -> list[str]:
    """Sample names in an output directory (``<sample>[.part<k>].pkl``)."""
    return sorted({_plot.sample_of(p)[0] for p in _pickles(outdir)})


def load_sample(outdir: str, sample: str) -> dict | None:
    """Merge every part pickle of exactly this sample into one result dict."""
    paths = [p for p in _pickles(outdir) if _plot.sample_of(p)[0] == sample]
    return merge_results(paths) if paths else None


def load_samples(outdir: str, only: list[str] | None = None) -> dict:
    """{sample: merged result dict} for an output directory."""
    out = {}
    for sample in samples_in(outdir):
        if only and sample not in only:
            continue
        data = load_sample(outdir, sample)
        if data:
            out[sample] = data
    return out


# ── histogram access ──────────────────────────────────────────────


def slice_1d(h, sample: str, hist_cfg: dict):
    """One sample's 1D slice with the config's rebin/flow applied, or None."""
    if _plot.is_2d_hist(h) or sample not in list(h.axes["dataset"]):
        return None
    return _plot.prep_1d(h[{"dataset": sample}], hist_cfg)


def normalized(sh):
    """Unit-area copy of a histogram, or None if it is empty."""
    total = sh.sum().value
    return sh * (1.0 / total) if total > 0 else None


def x_label(hist_cfg: dict, name: str) -> str:
    return hist_cfg.get("label", name)


def sample_label(sample_defs: dict, sample: str) -> str:
    return sample_defs.get(sample, {}).get("label", sample)


# ── output ────────────────────────────────────────────────────────


def save(fig, dest: str, name: str, formats=DEFAULT_FORMATS):
    save_figure(fig, dest, name, formats)


def write_gallery(dest: str, title: str, subtitle: str, ranked, score_label: str):
    """Thumbnail gallery of the ranked plots, in rank order with their scores."""
    os.makedirs(dest, exist_ok=True)
    return _plot.write_gallery(dest, ranked=ranked, title=title, subtitle=subtitle,
                               score_label=score_label)
