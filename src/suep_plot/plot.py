"""Plotting with mplhep – CMS-style figures from filled histograms."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

from .histograms import is_2d as _hist_is_2d

hep.style.use("CMS")


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
    """Merge multiple per-sample pickle files into one combined result."""
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
            merged.setdefault(key, {}).update(data.get(key, {}))
        for name, cfg in data.get("hist_defs", {}).items():
            merged.setdefault("hist_defs", {}).setdefault(name, cfg)
    return merged


def load_derived_plot_defs(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


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
):
    """Plot a single 1D histogram with CMS styling."""
    samples = list(h.axes["dataset"])
    if not samples:
        return

    signal_samples = []
    bkg_samples = []
    data_samples = []
    for s in samples:
        cfg = sample_defs.get(s, {})
        if cfg.get("is_data", False):
            data_samples.append(s)
        elif cfg.get("group") == "signal":
            signal_samples.append(s)
        else:
            bkg_samples.append(s)

    fig, ax = plt.subplots(figsize=(10, 8))

    if bkg_samples:
        bkg_hists = [h[{"dataset": s}] for s in bkg_samples]
        bkg_labels = [sample_defs.get(s, {}).get("label", s) for s in bkg_samples]
        bkg_colors = [sample_defs.get(s, {}).get("color", None) for s in bkg_samples]
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
        sh = h[{"dataset": s}]
        if normalize and sh.sum().value > 0:
            sh = sh * (1.0 / sh.sum().value)
        if solo_signal:
            hep.histplot(sh, ax=ax, histtype="fill", label=cfg.get("label", s),
                         color=cfg.get("color", "tab:blue"), alpha=0.7, edgecolor="black", linewidth=0.5)
        else:
            # sqrt(sum w^2) MC-stat errors; mplhep's default Poisson intervals
            # draw large upper limits on every empty bin of a weighted hist.
            hep.histplot(sh, ax=ax, histtype="step", label=cfg.get("label", s),
                         color=cfg.get("color", "red"), linewidth=2,
                         yerr=np.sqrt(sh.variances()))

    for s in data_samples:
        cfg = sample_defs.get(s, {})
        dh = h[{"dataset": s}]
        hep.histplot(
            dh,
            ax=ax,
            histtype="errorbar",
            label=cfg.get("label", data_label),
            color="black",
            marker="o",
            markersize=5,
        )

    ax.set_xlabel(hist_cfg.get("label", name))
    ax.set_ylabel("Events" if not normalize else "Normalized")
    if log_y:
        ax.set_yscale("log")
        if not normalize:
            ax.set_ylim(bottom=0.1)
    ax.legend(fontsize=12, loc="best")

    _cms_label(ax, lumi=lumi, has_data=bool(data_samples))

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2D histogram plotting ────────────────────────────────────────


def plot_histogram_2d(
    h: hist.Hist,
    hist_cfg: dict,
    sample_defs: dict,
    output_dir: str,
    name: str,
    *,
    lumi: float | None = None,
):
    """Plot a 2D histogram as colz (one figure per sample)."""
    samples = list(h.axes["dataset"])
    if not samples:
        return

    os.makedirs(output_dir, exist_ok=True)

    for s in samples:
        h2 = h[{"dataset": s}]
        fig, ax = plt.subplots(figsize=(10, 8))
        w = h2.view().value
        x_edges = h2.axes["x"].edges
        y_edges = h2.axes["y"].edges
        mesh = ax.pcolormesh(x_edges, y_edges, w.T, cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="Events")

        label = sample_defs.get(s, {}).get("label", s)
        ax.set_xlabel(hist_cfg.get("label_x", name))
        ax.set_ylabel(hist_cfg.get("label_y", ""))
        ax.set_title(label)

        _cms_label(ax, lumi=lumi)

        tag = f"{name}_{s}" if len(samples) > 1 else name
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(output_dir, f"{tag}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)


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


def plot_derived(
    derived_defs: dict,
    histograms: dict[str, hist.Hist],
    sample_defs: dict,
    output_dir: str,
    *,
    log_y: bool = False,
    lumi: float | None = None,
):
    """Produce profile, projection, efficiency, and ratio plots."""
    if not derived_defs:
        return

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for name, cfg in derived_defs.items():
        dtype = cfg["type"]

        if dtype in ("profile_x", "profile_y"):
            count += _plot_profile(name, cfg, histograms, sample_defs, output_dir, lumi=lumi)
        elif dtype in ("projection_x", "projection_y"):
            count += _plot_projection(name, cfg, histograms, sample_defs, output_dir,
                                       log_y=log_y, lumi=lumi)
        elif dtype == "efficiency":
            count += _plot_efficiency(name, cfg, histograms, sample_defs, output_dir, lumi=lumi)
        elif dtype == "ratio":
            count += _plot_ratio(name, cfg, histograms, sample_defs, output_dir, lumi=lumi)
        else:
            print(f"  WARNING: unknown derived type '{dtype}' for '{name}'")

    if count:
        print(f"Plotted {count} derived plot(s) to {output_dir}/")


def _plot_profile(name, cfg, histograms, sample_defs, output_dir, *, lumi=None):
    source = cfg["source"]
    if source not in histograms:
        print(f"  WARNING: source '{source}' not found for derived plot '{name}'")
        return 0

    h = histograms[source]
    axis = "x" if cfg["type"] == "profile_x" else "y"
    samples = list(h.axes["dataset"])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, s in enumerate(samples):
        h2 = h[{"dataset": s}]
        centers, mean, err, edges = _profile(h2, axis)
        label = sample_defs.get(s, {}).get("label", s)
        color = sample_defs.get(s, {}).get("color", colors[i % len(colors)])
        ax.errorbar(centers, mean, yerr=err, fmt="o", label=label,
                    color=color, markersize=4, capsize=2)

    profiled_axis = "y" if axis == "x" else "x"
    ax.set_xlabel(cfg.get("label_x", h.axes[axis].label))
    ax.set_ylabel(cfg.get("label_y", f"Mean {h.axes[profiled_axis].label}"))
    ax.legend(fontsize=12, loc="best")

    _cms_label(ax, lumi=lumi)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} (profile)")
    return 1


def _plot_projection(name, cfg, histograms, sample_defs, output_dir, *, log_y=False, lumi=None):
    source = cfg["source"]
    if source not in histograms:
        print(f"  WARNING: source '{source}' not found for derived plot '{name}'")
        return 0

    h = histograms[source]
    proj_axis = "x" if cfg["type"] == "projection_x" else "y"
    projected = h.project("dataset", proj_axis)

    hist_cfg = {
        "label": cfg.get("label_x", h.axes[proj_axis].label),
    }
    plot_histogram(projected, hist_cfg, sample_defs, output_dir, name,
                   log_y=log_y, lumi=lumi)
    print(f"  {name} (projection)")
    return 1


def _plot_efficiency(name, cfg, histograms, sample_defs, output_dir, *, lumi=None):
    num_name = cfg["numerator"]
    den_name = cfg["denominator"]
    for src in (num_name, den_name):
        if src not in histograms:
            print(f"  WARNING: histogram '{src}' not found for derived plot '{name}'")
            return 0

    h_num = histograms[num_name]
    h_den = histograms[den_name]
    samples = list(h_num.axes["dataset"])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, s in enumerate(samples):
        if s not in h_den.axes["dataset"]:
            continue
        passed = h_num[{"dataset": s}].view().value
        total = h_den[{"dataset": s}].view().value

        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(total > 0, passed / total, 0.0)

        lo, hi = _clopper_pearson(passed, total)
        err_lo = eff - lo
        err_hi = hi - eff

        centers = h_num.axes["x"].centers
        label = sample_defs.get(s, {}).get("label", s)
        color = sample_defs.get(s, {}).get("color", colors[i % len(colors)])
        ax.errorbar(centers, eff, yerr=[err_lo, err_hi], fmt="o", label=label,
                    color=color, markersize=4, capsize=2)

    ax.set_xlabel(cfg.get("label_x", h_num.axes["x"].label))
    ax.set_ylabel(cfg.get("label_y", "Efficiency"))
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=12, loc="best")

    _cms_label(ax, lumi=lumi)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} (efficiency)")
    return 1


def _plot_ratio(name, cfg, histograms, sample_defs, output_dir, *, lumi=None):
    num_name = cfg["numerator"]
    den_name = cfg["denominator"]
    for src in (num_name, den_name):
        if src not in histograms:
            print(f"  WARNING: histogram '{src}' not found for derived plot '{name}'")
            return 0

    h_num = histograms[num_name]
    h_den = histograms[den_name]
    samples = list(h_num.axes["dataset"])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, s in enumerate(samples):
        if s not in h_den.axes["dataset"]:
            continue
        nv = h_num[{"dataset": s}].view()
        dv = h_den[{"dataset": s}].view()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dv.value > 0, nv.value / dv.value, np.nan)
            err = np.where(dv.value > 0,
                           np.sqrt(nv.variance) / dv.value,
                           np.nan)

        centers = h_num.axes["x"].centers
        label = sample_defs.get(s, {}).get("label", s)
        color = sample_defs.get(s, {}).get("color", colors[i % len(colors)])
        ax.errorbar(centers, ratio, yerr=err, fmt="o", label=label,
                    color=color, markersize=4, capsize=2)

    ax.set_xlabel(cfg.get("label_x", h_num.axes["x"].label))
    ax.set_ylabel(cfg.get("label_y", "Ratio"))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=12, loc="best")

    _cms_label(ax, lumi=lumi)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} (ratio)")
    return 1


# ── Top-level entry point ────────────────────────────────────────


def plot_all(
    results_path: str | list[str],
    output_dir: str,
    normalize: bool = False,
    log_y: bool = False,
    lumi: float | None = None,
    config_dir: str | None = None,
):
    """Plot all histograms from per-sample pickle files.

    results_path can be a single .pkl file, a list of .pkl files,
    a directory containing .pkl files, or a mix of files and directories.
    Per-sample histograms are merged before plotting.  When *lumi* is given,
    MC samples are normalized to xs * lumi * 1000 / sumw.
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

    if lumi is not None:
        apply_xs_scaling(histograms, sample_defs, data.get("sumw", {}), lumi)

    n_1d = sum(1 for name in histograms if not _is_2d_hist(histograms[name]))
    n_2d = sum(1 for name in histograms if _is_2d_hist(histograms[name]))
    print(f"Plotting {n_1d} 1D + {n_2d} 2D histogram(s) to {output_dir}/")

    for name, h in histograms.items():
        cfg = hist_defs.get(name, {})
        if _is_2d_hist(h):
            plot_histogram_2d(h, cfg, sample_defs, output_dir, name, lumi=lumi)
        else:
            plot_histogram(h, cfg, sample_defs, output_dir, name,
                           normalize=normalize, log_y=log_y, lumi=lumi)
        print(f"  {name}")

    derived_defs = {}
    if config_dir:
        derived_defs = load_derived_plot_defs(os.path.join(config_dir, "derived_plots.yaml"))
    if derived_defs:
        print(f"Computing {len(derived_defs)} derived plot(s)...")
        plot_derived(derived_defs, histograms, sample_defs, output_dir,
                     log_y=log_y, lumi=lumi)

    print("Done.")
