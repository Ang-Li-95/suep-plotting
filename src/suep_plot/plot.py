"""Plotting with mplhep – CMS-style figures from filled histograms."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .histograms import is_2d as _hist_is_2d  # noqa: F401  (re-exported)

hep.style.use("CMS")

DEFAULT_FORMATS = ("png", "pdf")


# ── I/O helpers ───────────────────────────────────────────────────


def load_results(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_inputs(inputs: str | list[str]) -> list[str]:
    """Turn a mix of files and directories into a flat list of .pkl paths."""
    if isinstance(inputs, str):
        inputs = [inputs]
    out = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            out.extend(sorted(str(f) for f in p.glob("*.pkl")))
        elif p.is_file():
            out.append(str(p))
    return out


def merge_results(paths: list[str]) -> dict:
    """Merge multiple pickle files into one combined result.

    Histograms are summed, so the same sample may be spread over several
    pickles (e.g. ``<sample>.part<k>.pkl`` from a file-split Slurm run);
    its ``sumw``/``nevents``/cutflow counts are summed accordingly.
    """
    merged = None
    for p in paths:
        data = load_results(p)
        if merged is None:
            merged = data
            continue
        for name, h in data["histograms"].items():
            if name in merged["histograms"]:
                merged["histograms"][name] += h
            else:
                merged["histograms"][name] = h
        merged["samples"].update(data.get("samples", {}))
        for key in ("sumw", "nevents"):
            tgt = merged.setdefault(key, {})
            for s, v in data.get(key, {}).items():
                tgt[s] = tgt.get(s, 0) + v
        cf = merged.setdefault("cutflow", {})
        for s, flows in data.get("cutflow", {}).items():
            if s not in cf:
                cf[s] = flows
                continue
            for sel, counts in flows.items():
                if sel in cf[s]:
                    cf[s][sel] = {"raw": cf[s][sel]["raw"] + counts["raw"],
                                  "wtd": cf[s][sel]["wtd"] + counts["wtd"]}
                else:
                    cf[s][sel] = counts
        for name, cfg in data.get("hist_defs", {}).items():
            merged.setdefault("hist_defs", {}).setdefault(name, cfg)
    return merged


def load_derived_plot_defs(path: str) -> dict:
    from .config import load_config_file
    return load_config_file(path)


def _is_2d_hist(h: hist.Hist) -> bool:
    return len([a for a in h.axes if a.name != "dataset"]) == 2


def _cms_label(ax, *, lumi: float | None = None, has_data: bool = False):
    """CMS label: "Simulation Preliminary" for MC-only, "Preliminary" (+ lumi) for data."""
    # MDSNano samples are Run 3 -> 13.6 TeV (mplhep defaults to 13).
    hep.cms.label("Preliminary", data=has_data, lumi=lumi, com=13.6, ax=ax)


def apply_xs_scaling(histograms: dict, sample_defs: dict, sumw: dict, lumi: float) -> None:
    """Scale each MC sample to xs [pb] x lumi [/fb] using its summed genWeight.

    weight = xs * lumi * 1000 / sumw.  Data samples and MC samples without
    ``xs`` or a recorded ``sumw`` are left untouched (a warning is printed).
    """
    factors = {}
    for s, cfg in sample_defs.items():
        if cfg.get("is_data", False):
            continue
        xs = cfg.get("xs")
        sw = sumw.get(s, 0.0)
        if xs is None or not sw:
            missing = "xs" if xs is None else "sumw"
            print(f"  WARNING: MC sample '{s}' has no {missing}; left unscaled")
            continue
        factors[s] = float(xs) * lumi * 1000.0 / sw
    if not factors:
        return

    print("Normalizing MC to xs x lumi (weight = xs * lumi * 1000 / sumw):")
    for s, f in factors.items():
        print(f"  {s}: x {f:.4g}")

    for h in histograms.values():
        cats = list(h.axes["dataset"])
        view = h.view(flow=True)
        for s, f in factors.items():
            if s in cats:
                idx = cats.index(s)
                view["value"][idx] *= f
                view["variance"][idx] *= f * f


# ── Histogram styling helpers ─────────────────────────────────────


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


def _prep_1d(sh: hist.Hist, hist_cfg: dict) -> hist.Hist:
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


# ── Shared figure plumbing ────────────────────────────────────────


def _figure(ratio: bool = False):
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
    """(sample, 1D slice, label, color) per dataset, with the colour-cycle fallback."""
    colors = palette()
    for i, s in enumerate(h.axes["dataset"]):
        cfg = sample_defs.get(s, {})
        yield (s, h[{"dataset": s}], cfg.get("label", s),
               cfg.get("color", colors[i % len(colors)]))


def _require(histograms: dict, names, plot_name: str) -> bool:
    """Warn and return False if any source histogram is missing."""
    missing = [n for n in names if n not in histograms]
    if missing:
        print(f"  WARNING: source histogram(s) {missing} not found for "
              f"derived plot '{plot_name}'")
    return not missing


def _finish(fig, ax, output_dir: str, name: str, formats, *,
            lumi=None, has_data=False, xlabel=None, ylabel=None, legend=True):
    """Label, CMS-tag, save and close one figure -- the tail of every renderer."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=18, loc="best")
    _cms_label(ax, lumi=lumi, has_data=has_data)
    os.makedirs(output_dir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    return 1


# ── 1D histogram plotting ────────────────────────────────────────


def plot_histogram(
    h: hist.Hist,
    hist_cfg: dict,
    sample_defs: dict,
    output_dir: str,
    name: str,
    *,
    normalize: bool = False,
    log_y: bool = False,
    lumi: float | None = None,
    data_label: str = "Data",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    ratio: bool = True,
):
    """Plot a single 1D histogram with CMS styling.

    When both data and stacked backgrounds are present (and *ratio* is true),
    a Data/MC ratio panel with an MC-stat band is drawn underneath.
    Per-histogram config keys honoured at plot time: ``blind``, ``rebin``,
    ``flow: sum``, ``log_x``, ``log_y``.
    """
    samples = list(h.axes["dataset"])
    if not samples:
        return

    signal_samples, bkg_samples, data_samples = _split_samples(samples, sample_defs)
    if hist_cfg.get("blind"):
        data_samples = []

    bkg_hists = [_prep_1d(h[{"dataset": s}], hist_cfg) for s in bkg_samples]
    data_hists = [_prep_1d(h[{"dataset": s}], hist_cfg) for s in data_samples]

    want_ratio = bool(ratio and data_hists and bkg_hists)
    fig, ax, rax = _figure(ratio=want_ratio)

    if bkg_hists:
        bkg_labels = [sample_defs.get(s, {}).get("label", s) for s in bkg_samples]
        bkg_colors = [sample_defs.get(s, {}).get("color", None) for s in bkg_samples]
        if normalize:
            # Stacking unit-area histograms is meaningless, so in shape mode
            # each background is normalized on its own and overlaid as a step,
            # directly comparable with the signal curves below.
            bkg_ls = [sample_defs.get(s, {}).get("linestyle", "-") for s in bkg_samples]
            for bh, label, color, ls in zip(bkg_hists, bkg_labels, bkg_colors, bkg_ls):
                if bh.sum().value > 0:
                    bh = bh * (1.0 / bh.sum().value)
                hep.histplot(bh, ax=ax, histtype="step", label=label,
                             color=color, linewidth=2, linestyle=ls,
                             yerr=np.sqrt(bh.variances()))
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

    # Fill style only for a lone signal; overlaid signals read better as steps.
    solo_signal = not (bkg_samples or data_samples) and len(signal_samples) == 1

    for s in signal_samples:
        cfg = sample_defs.get(s, {})
        sh = _prep_1d(h[{"dataset": s}], hist_cfg)
        label = cfg.get("label", s)
        if normalize:
            if sh.sum().value > 0:
                sh = sh * (1.0 / sh.sum().value)
        else:
            scale = float(cfg.get("scale", 1.0) or 1.0)
            if scale != 1.0:
                sh = sh * scale
                label = f"{label} $\\times${scale:g}"
        if solo_signal:
            hep.histplot(sh, ax=ax, histtype="fill", label=label,
                         color=cfg.get("color", "tab:blue"), alpha=0.7, edgecolor="black", linewidth=0.5)
        else:
            # sqrt(sum w^2) MC-stat errors; mplhep's default Poisson intervals
            # draw large upper limits on every empty bin of a weighted hist.
            hep.histplot(sh, ax=ax, histtype="step", label=label,
                         color=cfg.get("color", "red"), linewidth=2,
                         linestyle=cfg.get("linestyle", "-"),
                         yerr=np.sqrt(sh.variances()))

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
        _draw_ratio_panel(rax, data_hists, bkg_hists)

    _finish(fig, ax, output_dir, name, formats, lumi=lumi,
            has_data=bool(data_samples), xlabel="" if rax is not None else None,
            ylabel="Normalized" if normalize else "Events")


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


# ── 2D histogram plotting ────────────────────────────────────────


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
    for s, h2, label, _color in _series(h, sample_defs):
        fig, ax, _rax = _figure()
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
                ylabel=hist_cfg.get("label_y", ""))


# ── Derived-plot helpers ──────────────────────────────────────────


def _profile(h2: hist.Hist, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a profile (weighted mean) from a 2D histogram.

    Returns (bin_centers, mean, mean_err, edges) of the profiled axis.
    """
    view = h2.view()
    w, w2 = view.value, view.variance

    if axis == "x":
        other_centers = h2.axes["y"].centers[np.newaxis, :]
        bin_centers = h2.axes["x"].centers
        edges = h2.axes["x"].edges
        sum_axis = 1
    else:
        other_centers = h2.axes["x"].centers[:, np.newaxis]
        bin_centers = h2.axes["y"].centers
        edges = h2.axes["y"].edges
        sum_axis = 0

    sumw = w.sum(axis=sum_axis)
    sumw2 = w2.sum(axis=sum_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(sumw > 0,
                        (w * other_centers).sum(axis=sum_axis) / sumw,
                        np.nan)
        spread = np.where(sumw > 0,
                          (w * other_centers ** 2).sum(axis=sum_axis) / sumw - mean ** 2,
                          np.nan)

    spread = np.maximum(spread, 0)
    # Effective entries for weighted fills: (sum w)^2 / sum w^2.
    n_eff = np.where(sumw2 > 0, sumw ** 2 / np.where(sumw2 > 0, sumw2, 1.0), 1.0)
    mean_err = np.sqrt(spread / n_eff)
    return bin_centers, mean, mean_err, edges


def _clopper_pearson(passed, total, level=0.6827):
    """Clopper-Pearson interval for binomial efficiency."""
    from scipy import stats
    alpha = 1.0 - level
    lo = np.where(passed > 0,
                  stats.beta.ppf(alpha / 2, passed, total - passed + 1),
                  0.0)
    hi = np.where(passed < total,
                  stats.beta.ppf(1 - alpha / 2, passed + 1, total - passed),
                  1.0)
    return lo, hi


def _plot_profile(name, cfg, histograms, sample_defs, output_dir, *,
                  lumi=None, formats=DEFAULT_FORMATS, **_):
    if not _require(histograms, [cfg["source"]], name):
        return 0
    h = histograms[cfg["source"]]
    axis = "x" if cfg["type"] == "profile_x" else "y"

    fig, ax, _rax = _figure()
    for _s, h2, label, color in _series(h, sample_defs):
        centers, mean, err, _edges = _profile(h2, axis)
        ax.errorbar(centers, mean, yerr=err, fmt="o", label=label,
                    color=color, markersize=4, capsize=2)

    profiled = "y" if axis == "x" else "x"
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h.axes[axis].label),
                   ylabel=cfg.get("label_y", f"Mean {h.axes[profiled].label}"))


def _plot_projection(name, cfg, histograms, sample_defs, output_dir, *,
                     log_y=False, lumi=None, formats=DEFAULT_FORMATS, **_):
    if not _require(histograms, [cfg["source"]], name):
        return 0
    h = histograms[cfg["source"]]
    proj_axis = "x" if cfg["type"] == "projection_x" else "y"

    plot_histogram(h.project("dataset", proj_axis),
                   {"label": cfg.get("label_x", h.axes[proj_axis].label)},
                   sample_defs, output_dir, name,
                   log_y=log_y, lumi=lumi, formats=formats)
    return 1


def _draw_spans(ax, spans):
    """Shade labelled x-axis regions (e.g. detector station positions).

    Each entry is ``{lo, hi, label}`` with an optional ``color``.
    """
    for span in spans:
        color = span.get("color", "gray")
        ax.axvspan(span["lo"], span["hi"], color=color, alpha=0.15, zorder=0,
                   linewidth=0)
        if span.get("label"):
            ax.text((span["lo"] + span["hi"]) / 2, 0.98, span["label"],
                    transform=ax.get_xaxis_transform(), rotation=90,
                    ha="center", va="top", fontsize=11, color="dimgray")


def _plot_efficiency(name, cfg, histograms, sample_defs, output_dir, *,
                     lumi=None, formats=DEFAULT_FORMATS, **_):
    num_name, den_name = cfg["numerator"], cfg["denominator"]
    if not _require(histograms, (num_name, den_name), name):
        return 0
    h_num, h_den = histograms[num_name], histograms[den_name]
    if _is_2d_hist(h_num):
        return _plot_efficiency_2d(name, cfg, h_num, h_den, sample_defs,
                                   output_dir, lumi=lumi, formats=formats)

    fig, ax, _rax = _figure()
    centers = h_num.axes["x"].centers
    for s, hn, label, color in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        passed = hn.view().value
        total = h_den[{"dataset": s}].view().value

        filled = total > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(filled, passed / total, 0.0)
        lo, hi = _clopper_pearson(passed, total)

        ax.errorbar(centers[filled], eff[filled],
                    yerr=[(eff - lo)[filled], (hi - eff)[filled]], fmt="o",
                    label=label, color=color, markersize=4, capsize=2)

    if cfg.get("spans"):
        _draw_spans(ax, cfg["spans"])
    ax.set_ylim(-0.05, 1.15)
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h_num.axes["x"].label),
                   ylabel=cfg.get("label_y", "Efficiency"))


def _plot_ratio(name, cfg, histograms, sample_defs, output_dir, *,
                lumi=None, formats=DEFAULT_FORMATS, **_):
    num_name, den_name = cfg["numerator"], cfg["denominator"]
    if not _require(histograms, (num_name, den_name), name):
        return 0
    h_num, h_den = histograms[num_name], histograms[den_name]

    fig, ax, _rax = _figure()
    centers = h_num.axes["x"].centers
    for s, hn, label, color in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        nv, dv = hn.view(), h_den[{"dataset": s}].view()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dv.value > 0, nv.value / dv.value, np.nan)
            err = np.where(dv.value > 0, np.sqrt(nv.variance) / dv.value, np.nan)
        ax.errorbar(centers, ratio, yerr=err, fmt="o", label=label,
                    color=color, markersize=4, capsize=2)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h_num.axes["x"].label),
                   ylabel=cfg.get("label_y", "Ratio"))


def _plot_sig_vs_bkg(name, cfg, histograms, sample_defs, output_dir, *,
                     log_y=False, lumi=None, formats=DEFAULT_FORMATS, **_):
    """Overlay a different histogram for signal and for background samples.

    Signal samples are drawn from the ``signal`` histogram and background
    samples from the ``background`` one, so e.g. gen-matched signal clusters
    can be compared with the full cluster population of a truth-less
    background sample without refilling either.  Curves are normalized to
    unit area by default (``normalize: false`` keeps raw yields), since the
    two source histograms generally hold different populations.
    """
    sig_name, bkg_name = cfg["signal"], cfg["background"]
    if not _require(histograms, (sig_name, bkg_name), name):
        return 0
    h_sig, h_bkg = histograms[sig_name], histograms[bkg_name]
    normalize = cfg.get("normalize", True)

    # Each sample is drawn from the histogram matching its own group.
    sig_samples, bkg_samples, data_samples = _split_samples(
        list(h_sig.axes["dataset"]), sample_defs)
    entries = [(s, h_sig) for s in sig_samples]
    entries += [(s, h_bkg) for s in bkg_samples + data_samples
                if s in h_bkg.axes["dataset"]]
    if not entries:
        print(f"  WARNING: no samples to draw for derived plot '{name}'")
        return 0

    fig, ax, _rax = _figure()
    colors = palette()

    drawn = 0
    for i, (s, h) in enumerate(entries):
        sh = _prep_1d(h[{"dataset": s}], cfg)
        if sh.sum().value <= 0:
            continue
        if normalize:
            sh = sh * (1.0 / sh.sum().value)
        cfg_s = sample_defs.get(s, {})
        hep.histplot(sh, ax=ax, histtype="step", linewidth=2,
                     label=cfg_s.get("label", s),
                     color=cfg_s.get("color", colors[i % len(colors)]),
                     yerr=np.sqrt(sh.variances()))
        drawn += 1

    if not drawn:
        plt.close(fig)
        print(f"  WARNING: all source histograms empty for derived plot '{name}'")
        return 0

    if log_y or cfg.get("log_y"):
        ax.set_yscale("log")
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h_sig.axes["x"].label),
                   ylabel=cfg.get("label_y",
                                  "Normalized" if normalize else "Events"))


def _plot_efficiency_2d(name, cfg, h_num, h_den, sample_defs, output_dir, *,
                        lumi=None, formats=DEFAULT_FORMATS):
    """Efficiency map from a 2D numerator/denominator pair (one figure per
    sample).  Bins with an empty denominator are left blank; the color scale
    is fixed to [0, 1].  ``label_z`` sets the colorbar label."""
    x_edges, y_edges = h_num.axes["x"].edges, h_num.axes["y"].edges
    multi = len(h_num.axes["dataset"]) > 1

    plotted = 0
    for s, hn, label, _color in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            total = h_den[{"dataset": s}].view().value
            eff = np.where(total > 0, hn.view().value / total, np.nan)

        fig, ax, _rax = _figure()
        mesh = ax.pcolormesh(x_edges, y_edges, eff.T, cmap="viridis",
                             vmin=0.0, vmax=1.0)
        fig.colorbar(mesh, ax=ax, label=cfg.get("label_z", "Efficiency"))
        # Sample tag inside the axes; a centered title collides with the CMS label.
        ax.text(0.97, 0.97, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=16)
        _finish(fig, ax, output_dir, f"{name}_{s}" if multi else name, formats,
                lumi=lumi, legend=False,
                xlabel=cfg.get("label_x", h_num.axes["x"].label),
                ylabel=cfg.get("label_y", h_num.axes["y"].label))
        plotted += 1
    return 1 if plotted else 0


# The derived-plot types, each with the config keys naming its source
# histograms.  One table, read both by plot_all (to decide which histograms a
# task needs) and by _plot_derived_one (to dispatch it) -- the two used to
# carry separate copies, which is how a new type could be added to one and not
# the other.
DERIVED_TYPES = {
    "profile_x":    (_plot_profile, ("source",)),
    "profile_y":    (_plot_profile, ("source",)),
    "projection_x": (_plot_projection, ("source",)),
    "projection_y": (_plot_projection, ("source",)),
    "efficiency":   (_plot_efficiency, ("numerator", "denominator")),
    "ratio":        (_plot_ratio, ("numerator", "denominator")),
    "sig_vs_bkg":   (_plot_sig_vs_bkg, ("signal", "background")),
}


def derived_sources(name: str, cfg: dict) -> list[str] | None:
    """The histograms a derived plot reads, or None if its type is unknown."""
    entry = DERIVED_TYPES.get(cfg.get("type", ""))
    if entry is None:
        print(f"  WARNING: unknown derived type '{cfg.get('type', '')}' for '{name}'")
        return None
    return [cfg.get(key) for key in entry[1]]


def _plot_derived_one(name, cfg, histograms, sample_defs, output_dir, *,
                      log_y=False, lumi=None, formats=DEFAULT_FORMATS) -> int:
    """Dispatch a single derived-plot definition to its renderer."""
    entry = DERIVED_TYPES.get(cfg.get("type", ""))
    if entry is None:
        print(f"  WARNING: unknown derived type '{cfg.get('type', '')}' for '{name}'")
        return 0
    return entry[0](name, cfg, histograms, sample_defs, output_dir,
                    log_y=log_y, lumi=lumi, formats=formats)


# ── Cutflow table ────────────────────────────────────────────────


def write_cutflow(cutflow: dict, output_dir: str):
    """Write per-sample cutflow tables (raw and weighted counts) as txt + csv.

    ``cutflow`` maps sample -> selection -> {"raw": int, "wtd": float}.
    Counts are independent per selection, not sequential.
    """
    if not cutflow:
        return
    os.makedirs(output_dir, exist_ok=True)

    sel_names: list[str] = []
    for flows in cutflow.values():
        for sel in flows:
            if sel not in sel_names:
                sel_names.append(sel)

    txt_path = os.path.join(output_dir, "cutflow.txt")
    csv_path = os.path.join(output_dir, "cutflow.csv")

    with open(txt_path, "w") as f:
        f.write("Cutflow (independent counts per selection; 'total' = all processed events)\n")
        for sample, flows in cutflow.items():
            f.write(f"\n{sample}\n")
            f.write(f"  {'selection':<28}{'raw':>12}{'weighted':>16}\n")
            for sel in sel_names:
                if sel not in flows:
                    continue
                c = flows[sel]
                f.write(f"  {sel:<28}{c['raw']:>12,}{c['wtd']:>16.4g}\n")

    with open(csv_path, "w") as f:
        f.write("sample,selection,raw,weighted\n")
        for sample, flows in cutflow.items():
            for sel in sel_names:
                if sel in flows:
                    c = flows[sel]
                    f.write(f"{sample},{sel},{c['raw']},{c['wtd']:.6g}\n")

    print(f"Wrote cutflow tables: {txt_path}, {csv_path}")


# ── ROOT export (for combine / further processing) ───────────────


def export_root(histograms: dict[str, hist.Hist], path: str):
    """Write every histogram x dataset slice to a ROOT file (TH1D/TH2D).

    Layout: one directory per histogram, one key per sample
    (``<hist>/<sample>``), ready for combine datacards or ROOT-based tooling.
    """
    import uproot

    n = 0
    with uproot.recreate(path) as f:
        for name, h in histograms.items():
            for s in h.axes["dataset"]:
                f[f"{name}/{s}"] = h[{"dataset": s}]
                n += 1
    print(f"Wrote {n} histogram(s) to {path}")


# ── HTML gallery ─────────────────────────────────────────────────


def write_gallery(output_dir: str) -> str | None:
    """Write an index.html thumbnail gallery of every PNG in *output_dir*.

    Each thumbnail links to the PDF when one exists (else the PNG), so the
    whole plot set can be browsed from a single page over ssh/web.
    """
    out = Path(output_dir)
    pngs = sorted(out.glob("*.png"))
    if not pngs:
        return None

    cards = []
    for png in pngs:
        pdf = png.with_suffix(".pdf")
        href = pdf.name if pdf.exists() else png.name
        cards.append(
            f'<figure><a href="{href}"><img src="{png.name}" loading="lazy" '
            f'alt="{png.stem}"></a><figcaption>{png.stem}</figcaption></figure>'
        )
    cutflow_link = ('<p><a href="cutflow.txt">cutflow.txt</a> · '
                    '<a href="cutflow.csv">cutflow.csv</a></p>'
                    if (out / "cutflow.txt").exists() else "")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>suep-plot: {out.name}</title>
<style>
 body {{ font-family: sans-serif; margin: 1.5em; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1em; }}
 figure {{ margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 6px; }}
 img {{ width: 100%; height: auto; }}
 figcaption {{ text-align: center; font-size: 0.85em; padding-top: 4px;
               font-family: monospace; word-break: break-all; }}
</style></head><body>
<h1>{out.name} ({len(pngs)} plots)</h1>
{cutflow_link}
<div class="grid">
{chr(10).join(cards)}
</div></body></html>
"""
    path = out / "index.html"
    path.write_text(html)
    return str(path)


# ── Parallel rendering ───────────────────────────────────────────


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
            _plot_derived_one(name, cfg, hists, common["sample_defs"], common["output_dir"],
                              log_y=common["log_y"], lumi=common["lumi"],
                              formats=common["formats"])
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


# ── Top-level entry point ────────────────────────────────────────


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
    blind, rebin, flow, log_x, log_y, log_z ...) override the ones stored in
    the pickles, so styling can be iterated without reprocessing.
    """
    pkl_files = _resolve_inputs(results_path)
    if not pkl_files:
        print("No .pkl files found.")
        return
    print(f"Loading {len(pkl_files)} sample file(s):")
    for f in pkl_files:
        print(f"  {f}")
    data = merge_results(pkl_files)

    histograms = data["histograms"]
    sample_defs = data.get("samples", {})
    hist_defs = data.get("hist_defs", {})

    if config_dir:
        fresh_path = Path(config_dir) / "histograms.yaml"
        if fresh_path.exists():
            from .histograms import load_histogram_defs
            for name, cfg in load_histogram_defs(fresh_path).items():
                hist_defs[name] = {**hist_defs.get(name, {}), **cfg}

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
        kind = "2d" if _is_2d_hist(h) else "1d"
        tasks.append((kind, name, h, hist_defs.get(name, {})))
    n_1d = sum(1 for t in tasks if t[0] == "1d")
    n_2d = sum(1 for t in tasks if t[0] == "2d")

    derived_defs = {}
    if config_dir:
        derived_defs = load_derived_plot_defs(os.path.join(config_dir, "derived_plots.yaml"))
    for name, cfg in derived_defs.items():
        needed = derived_sources(name, cfg)
        if needed is None or not _require(histograms, needed, name):
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

    gallery = write_gallery(output_dir)
    if gallery:
        print(f"Gallery: {gallery}")

    if failed:
        print(f"Done ({failed} plot(s) failed).")
    else:
        print("Done.")
