"""Shared suep-plot styling for the standalone comparison scripts.

The comparison scripts read the same pickles ``suep-plot`` reads, so they reuse
its helpers directly: mplhep CMS style, the plot-time transforms declared in
``histograms.yaml`` (rebin / flow folding / log axes), the per-sample labels and
colors from ``samples.yaml``, and png+pdf output with an HTML thumbnail gallery.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

from suep_plot.plot import (  # noqa: E402
    DEFAULT_FORMATS,
    _cms_label,
    _is_2d_hist,
    _prep_1d,
    merge_results,
)

hep.style.use("CMS")

# Figure geometry copied from suep_plot.plot so the two sets of figures can sit
# side by side in the same talk.
FIGSIZE = (10, 8)
FIGSIZE_RATIO = (10, 10)
LEGEND_FONTSIZE = 18
DPI = 150


def cms_label(ax, lumi=None):
    """CMS 'Preliminary' header at 13.6 TeV (simulation-only by default)."""
    _cms_label(ax, lumi=lumi)


def palette() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── loading ───────────────────────────────────────────────────────


def samples_in(outdir: str) -> list[str]:
    """Sample names in an output directory (``<sample>[.part<k>].pkl``)."""
    return sorted({os.path.basename(p).split(".")[0]
                   for p in glob.glob(f"{outdir}/*.pkl")})


def load_sample(outdir: str, sample: str) -> dict | None:
    """Merge every part pickle of one sample into a single result dict."""
    paths = sorted(glob.glob(f"{outdir}/{sample}*.pkl"))
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
    if _is_2d_hist(h) or sample not in list(h.axes["dataset"]):
        return None
    return _prep_1d(h[{"dataset": sample}], hist_cfg)


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
    os.makedirs(dest, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(dest, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_gallery(dest: str, title: str, subtitle: str, ranked, score_label: str):
    """Thumbnail gallery of the ranked plots, styled like suep-plot's index.html.

    ``ranked`` is [(plot_name, score)] in the order to display; each thumbnail
    links to the PDF when one exists.
    """
    out = Path(dest)
    cards = []
    for name, score in ranked:
        pdf = out / f"{name}.pdf"
        href = pdf.name if pdf.exists() else f"{name}.png"
        cards.append(
            f'<figure><a href="{href}"><img src="{name}.png" loading="lazy" '
            f'alt="{name}"></a><figcaption>{name}<br>'
            f'<span class="score">{score_label} {score:.3f}</span></figcaption></figure>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>suep-plot: {out.name}</title>
<style>
 body {{ font-family: sans-serif; margin: 1.5em; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1em; }}
 figure {{ margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 6px; }}
 img {{ width: 100%; height: auto; }}
 figcaption {{ text-align: center; font-size: 0.85em; padding-top: 4px;
               font-family: monospace; word-break: break-all; }}
 .score {{ color: #666; }}
</style></head><body>
<h1>{title} ({len(ranked)} plots)</h1>
<p>{subtitle}</p>
<div class="grid">
{chr(10).join(cards)}
</div></body></html>
"""
    path = out / "index.html"
    path.write_text(html)
    return str(path)
