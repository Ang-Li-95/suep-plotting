"""Drawing with mplhep: CMS-style 1D and 2D histogram figures.

The figure helpers here (:func:`figure`, :func:`cms_label`, :func:`prep_1d`,
:func:`save_figure`, ...) are shared by the derived plots
(:mod:`suep_plot.derived`) and by the comparison scripts in ``scripts/``.
"""

from __future__ import annotations

import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")


DEFAULT_FORMATS = ("png", "pdf")


# Point size for the errorbar plots (efficiency / profile / ratio): large
# enough that the marker shape reads at a glance, since shape is what
# separates two samples sharing a colour.
MARKERSIZE = 8


# ── histogram helpers ─────────────────────────────────────────────


def is_2d_hist(h: hist.Hist) -> bool:
    return len([a for a in h.axes if a.name != "dataset"]) == 2


def _fold_flow(sh: hist.Hist) -> hist.Hist:
    """Fold under/overflow into the first/last visible bin of a 1D histogram."""
    sh = sh.copy()
    v = sh.view(flow=True)
    for field in ("value", "variance"):
        v[field][1] += v[field][0]
        v[field][0] = 0.0
        v[field][-2] += v[field][-1]
        v[field][-1] = 0.0
    return sh


def prep_1d(sh: hist.Hist, hist_cfg: dict) -> hist.Hist:
    """Apply plot-time transforms (rebin, overflow folding) to a 1D slice.

    Under/overflow is folded into the first/last visible bin by default, so
    every entry is on the canvas and normalized curves are normalized over the
    same population even when samples spill out of the axis range by different
    amounts.  ``flow: none`` in the histogram config opts out and drops the
    out-of-range entries instead.
    """
    rebin = int(hist_cfg.get("rebin") or 0)
    if rebin > 1:
        sh = sh[:: hist.rebin(rebin)]
    if str(hist_cfg.get("flow", "sum")).lower() not in ("none", "omit", "drop"):
        sh = _fold_flow(sh)
    return sh


def _split_samples(samples, sample_defs):
    signal, background, data = [], [], []
    for s in samples:
        cfg = sample_defs.get(s, {})
        if cfg.get("is_data", False):
            data.append(s)
        elif cfg.get("group") == "signal":
            signal.append(s)
        else:
            background.append(s)
    return signal, background, data


# ── figure plumbing ───────────────────────────────────────────────


def cms_label(ax, *, lumi: float | None = None, has_data: bool = False):
    """CMS label: "Simulation Preliminary" for MC-only, "Preliminary" (+ lumi) for data."""
    # MDSNano samples are Run 3 -> 13.6 TeV (mplhep defaults to 13).
    hep.cms.label("Preliminary", data=has_data, lumi=lumi, com=13.6, ax=ax)


def figure(ratio: bool = False):
    """A CMS-sized figure: one panel, or a main panel with a ratio panel."""
    if ratio:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        return fig, ax, rax
    fig, ax = plt.subplots(figsize=(10, 8))
    return fig, ax, None


def palette() -> list[str]:
    """The active colour cycle -- the fallback when a sample defines no colour."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _series(h: hist.Hist, sample_defs: dict):
    """(sample, 1D slice, label, color, linestyle, marker) per dataset.

    The colour falls back to the cycle.  Style and marker matter because
    samples that differ in only one parameter share a colour on purpose (the
    signal grid gives both temperatures of one mDark the same colour), so they
    are the only thing telling them apart in a point plot.
    """
    colors = palette()
    for i, s in enumerate(h.axes["dataset"]):
        cfg = sample_defs.get(s, {})
        yield (s, h[{"dataset": s}], cfg.get("label", s),
               cfg.get("color", colors[i % len(colors)]),
               cfg.get("linestyle", "-"), cfg.get("marker", "o"))


def _finish(fig, ax, output_dir: str, name: str, formats, *,
            lumi=None, has_data=False, xlabel=None, ylabel=None, legend=True,
            annotation=None):
    """Label, CMS-tag, save and close one figure -- the tail of every renderer.

    *annotation* (a region histogram's "CSC · muon-isolated") becomes the
    legend title, or a corner text when there is no legend.  A *name* with
    slashes (``csc/iso_muon/csc_cluster_size``) is saved into subfolders.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=18, loc="best", title=annotation, title_fontsize=16)
    elif annotation:
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha="left",
                va="top", fontsize=16)
    cms_label(ax, lumi=lumi, has_data=has_data)
    save_figure(fig, output_dir, name, formats)
    return 1


def save_figure(fig, output_dir: str, name: str, formats=DEFAULT_FORMATS) -> None:
    """Save *fig* as ``<output_dir>/<name>.<ext>`` per format, then close it.

    A *name* with slashes goes into subfolders.  Shared with scripts/.
    """
    base = os.path.join(output_dir, name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 1D histograms ─────────────────────────────────────────────────


def _unit_area(h: hist.Hist) -> hist.Hist:
    """*h* scaled to unit area (unchanged when it is empty)."""
    total = h.sum().value
    return h * (1.0 / total) if total > 0 else h


def _total(hists: list[hist.Hist]) -> hist.Hist:
    out = hists[0].copy()
    for h in hists[1:]:
        out += h
    return out


STAT_LABEL = "Stat. unc."


def draw_uncertainty_band(ax, h: hist.Hist, *, color="gray", hatch=None,
                          alpha=0.3, label=None) -> None:
    """The +-1 sigma MC-stat band (sqrt of the summed squared weights) of *h*.

    Drawn as a step band over the bins: translucent in *color*, or hatched in
    *color* with no fill when *hatch* is given (the stack total, and filled
    curves, whose own fill would hide a translucent band).  Shared with
    scripts/.
    """
    edges = h.axes[0].edges
    val = h.values()
    err = np.sqrt(h.variances())
    hi = np.r_[val + err, (val + err)[-1]]
    lo = np.r_[val - err, (val - err)[-1]]
    # A bin of one weighted entry has err == val up to rounding, leaving a
    # ~1e-20 remainder that a log axis would stretch down to: call it zero
    # (non-positive values do not take part in log autoscaling).
    lo = np.where(lo > 1e-9 * hi, lo, 0.0)
    if hatch:
        ax.fill_between(edges, lo, hi, step="post", facecolor="none", edgecolor=color,
                        hatch=hatch, linewidth=0, label=label)
    else:
        ax.fill_between(edges, lo, hi, step="post", color=color, alpha=alpha,
                        linewidth=0, label=label)


def _step_with_band(ax, h: hist.Hist, label, color, linestyle) -> None:
    """A step curve with its MC-stat band in the same colour.

    The band is sqrt(sum w^2), not mplhep's default Poisson interval, which
    draws large upper limits on every empty bin of a weighted histogram.
    """
    artists = hep.histplot(h, ax=ax, histtype="step", label=label, color=color,
                           linewidth=2, linestyle=linestyle, yerr=False)
    if color is None and artists:
        # a colourless sample took the next colour of the cycle; match it
        color = artists[0].stairs.get_edgecolor()
    draw_uncertainty_band(ax, h, color=color)


def plot_histogram(
    h: hist.Hist,
    hist_cfg: dict,
    sample_defs: dict,
    output_dir: str,
    name: str,
    *,
    normalize: bool = False,
    unit_stack: bool = False,
    log_y: bool = False,
    lumi: float | None = None,
    data_label: str = "Data",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    ratio: bool = True,
):
    """Plot a single 1D histogram with CMS styling.

    When both data and stacked backgrounds are present (and *ratio* is true),
    a Data/MC ratio panel with an MC-stat band is drawn underneath.
    With *normalize* every sample -- signal, background and data -- is drawn
    at unit area, and the ratio panel compares the shapes: all data over all
    backgrounds, each total at unit area.
    *unit_stack* is *normalize* with the backgrounds kept as a stack: they are
    all divided by the integral of their sum, so the stack has unit area and
    its composition stays visible (meaningful once the samples carry their
    relative normalization, i.e. with a luminosity).
    Per-histogram config keys honoured at plot time: ``blind``, ``rebin``,
    ``flow: sum``, ``log_x``, ``log_y``.
    """
    samples = list(h.axes["dataset"])
    if not samples:
        return

    signal_samples, bkg_samples, data_samples = _split_samples(samples, sample_defs)
    if hist_cfg.get("blind"):
        data_samples = []

    bkg_hists = [prep_1d(h[{"dataset": s}], hist_cfg) for s in bkg_samples]
    # Smallest yield at the bottom of the stack: on a log axis a small process
    # drawn on top of a large one is a sliver nobody can read.  (Stable, so
    # equal yields keep the config order.)
    order = sorted(range(len(bkg_samples)), key=lambda i: bkg_hists[i].sum().value)
    bkg_samples = [bkg_samples[i] for i in order]
    bkg_hists = [bkg_hists[i] for i in order]
    data_hists = [prep_1d(h[{"dataset": s}], hist_cfg) for s in data_samples]

    want_ratio = bool(ratio and data_hists and bkg_hists)
    fig, ax, rax = figure(ratio=want_ratio)
    normalize = normalize or unit_stack
    ratio_data, ratio_bkg = data_hists, bkg_hists
    if normalize:
        if want_ratio:
            ratio_data = [_unit_area(_total(data_hists))]
            ratio_bkg = [_unit_area(_total(bkg_hists))]
        data_hists = [_unit_area(dh) for dh in data_hists]

    if bkg_hists:
        bkg_labels = [sample_defs.get(s, {}).get("label", s) for s in bkg_samples]
        bkg_colors = [sample_defs.get(s, {}).get("color", None) for s in bkg_samples]
        if unit_stack:
            total = _total(bkg_hists).sum().value
            if total > 0:
                bkg_hists = [bh * (1.0 / total) for bh in bkg_hists]
        if normalize and not unit_stack:
            # Stacking unit-area histograms is meaningless, so in shape mode
            # each background is normalized on its own and overlaid as a step,
            # directly comparable with the signal curves below.
            bkg_ls = [sample_defs.get(s, {}).get("linestyle", "-") for s in bkg_samples]
            for bh, label, color, ls in zip(bkg_hists, bkg_labels, bkg_colors, bkg_ls):
                bh = _unit_area(bh)
                _step_with_band(ax, bh, label, color, ls)
        else:
            hep.histplot(
                bkg_hists,
                ax=ax,
                stack=True,
                histtype="fill",
                label=bkg_labels,
                color=bkg_colors,
                edgecolor="black",
                linewidth=0.5,
            )
            draw_uncertainty_band(ax, _total(bkg_hists), hatch="////",
                                  label=STAT_LABEL)

    # Fill style only for a lone signal; overlaid signals read better as steps.
    solo_signal = not (bkg_samples or data_samples) and len(signal_samples) == 1

    for s in signal_samples:
        cfg = sample_defs.get(s, {})
        sh = prep_1d(h[{"dataset": s}], hist_cfg)
        label = cfg.get("label", s)
        if normalize:
            sh = _unit_area(sh)
        else:
            scale = float(cfg.get("scale", 1.0) or 1.0)
            if scale != 1.0:
                sh = sh * scale
                label = f"{label} $\\times${scale:g}"
        if solo_signal:
            hep.histplot(sh, ax=ax, histtype="fill", label=label,
                         color=cfg.get("color", "tab:blue"), alpha=0.7, edgecolor="black", linewidth=0.5)
            draw_uncertainty_band(ax, sh, hatch="////", label=STAT_LABEL)
        else:
            _step_with_band(ax, sh, label, cfg.get("color", "red"),
                            cfg.get("linestyle", "-"))

    for s, dh in zip(data_samples, data_hists):
        cfg = sample_defs.get(s, {})
        hep.histplot(
            dh,
            ax=ax,
            histtype="errorbar",
            label=cfg.get("label", data_label),
            color="black",
            marker="o",
            markersize=5,
            # Scaled counts are no longer Poisson: sqrt(sum w^2), as for MC.
            **({"yerr": np.sqrt(dh.variances())} if normalize else {}),
        )

    # With a ratio panel the x label belongs under it, not under the main axes.
    (rax if rax is not None else ax).set_xlabel(hist_cfg.get("label", name))
    if log_y or hist_cfg.get("log_y"):
        ax.set_yscale("log")
        if not normalize:
            ax.set_ylim(bottom=0.1)
    if hist_cfg.get("log_x"):
        ax.set_xscale("log")
    if want_ratio:
        _draw_ratio_panel(rax, ratio_data, ratio_bkg)

    _finish(fig, ax, output_dir, name, formats, lumi=lumi,
            has_data=bool(data_samples), xlabel="" if rax is not None else None,
            ylabel="Normalized" if normalize else "Events",
            annotation=hist_cfg.get("annotation"))


def _draw_ratio_panel(rax, data_hists: list[hist.Hist], bkg_hists: list[hist.Hist]):
    """Data / total-background ratio with an MC-stat band around 1."""
    tot_val = np.sum([bh.view().value for bh in bkg_hists], axis=0)
    tot_var = np.sum([bh.view().variance for bh in bkg_hists], axis=0)
    dat_val = np.sum([dh.view().value for dh in data_hists], axis=0)
    dat_var = np.sum([dh.view().variance for dh in data_hists], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(tot_val > 0, dat_val / tot_val, np.nan)
        rerr = np.where(tot_val > 0, np.sqrt(dat_var) / tot_val, np.nan)
        band = np.where(tot_val > 0, np.sqrt(tot_var) / tot_val, 0.0)

    edges = bkg_hists[0].axes[0].edges
    centers = bkg_hists[0].axes[0].centers
    rax.fill_between(edges, np.r_[1 - band, (1 - band)[-1]], np.r_[1 + band, (1 + band)[-1]],
                     step="post", color="gray", alpha=0.35, linewidth=0)
    rax.errorbar(centers, r, yerr=rerr, fmt="o", color="black", markersize=5)
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    rax.set_ylabel("Data / MC")
    rax.set_ylim(0.5, 1.5)


# ── 2D histograms ─────────────────────────────────────────────────


def plot_histogram_2d(
    h: hist.Hist,
    hist_cfg: dict,
    sample_defs: dict,
    output_dir: str,
    name: str,
    *,
    lumi: float | None = None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
):
    """Plot a 2D histogram as colz (one figure per sample).

    ``log_z: true`` in the histogram config switches to a log color scale.
    """
    multi = len(h.axes["dataset"]) > 1
    for s, h2, label, _color, _ls, _mk in _series(h, sample_defs):
        fig, ax, _rax = figure()
        w = h2.view().value
        norm = None
        if hist_cfg.get("log_z") and (w > 0).any():
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=w[w > 0].min(), vmax=w.max())
        mesh = ax.pcolormesh(h2.axes["x"].edges, h2.axes["y"].edges, w.T,
                             cmap="viridis", norm=norm)
        fig.colorbar(mesh, ax=ax, label="Events")
        # Sample tag inside the axes; a centered title collides with the CMS label.
        ax.text(0.97, 0.97, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=16)
        _finish(fig, ax, output_dir, f"{name}_{s}" if multi else name, formats,
                lumi=lumi, legend=False,
                xlabel=hist_cfg.get("label_x", name),
                ylabel=hist_cfg.get("label_y", ""),
                annotation=hist_cfg.get("annotation"))
