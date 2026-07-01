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
    return merged


def load_derived_plot_defs(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_2d_hist(h: hist.Hist) -> bool:
    return len([a for a in h.axes if a.name != "dataset"]) == 2


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

    has_bkg_or_data = bool(bkg_samples) or bool(data_samples)

    for s in signal_samples:
        cfg = sample_defs.get(s, {})
        sh = h[{"dataset": s}]
        if normalize and sh.sum().value > 0:
            sh = sh * (1.0 / sh.sum().value)
        if has_bkg_or_data:
            hep.histplot(sh, ax=ax, histtype="step", label=cfg.get("label", s),
                         color=cfg.get("color", "red"), linewidth=2)
        else:
            hep.histplot(sh, ax=ax, histtype="fill", label=cfg.get("label", s),
                         color=cfg.get("color", "tab:blue"), alpha=0.7, edgecolor="black", linewidth=0.5)

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
        ax.set_ylim(bottom=0.1)
    ax.legend(fontsize=12, loc="best")

    if lumi is not None:
        hep.cms.label("Preliminary", data=bool(data_samples), lumi=lumi, year="2024", ax=ax)
    else:
        hep.cms.label("Preliminary", data=False, ax=ax)

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

        if lumi is not None:
            hep.cms.label("Preliminary", data=False, lumi=lumi, year="2024", ax=ax)
        else:
            hep.cms.label("Preliminary", data=False, ax=ax)

        tag = f"{name}_{s}" if len(samples) > 1 else name
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(output_dir, f"{tag}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Derived-plot helpers ──────────────────────────────────────────


def _profile(h2: hist.Hist, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a profile (weighted mean) from a 2D histogram.

    Returns (bin_centers, mean, mean_err, edges) of the profiled axis.
    """
    w = h2.view().value

    with np.errstate(divide="ignore", invalid="ignore"):
        if axis == "x":
            other_centers = h2.axes["y"].centers
            bin_centers = h2.axes["x"].centers
            edges = h2.axes["x"].edges
            sumw = w.sum(axis=1)
            mean = np.where(sumw > 0,
                            (w * other_centers[np.newaxis, :]).sum(axis=1) / sumw,
                            np.nan)
            spread = np.where(sumw > 0,
                              (w * other_centers[np.newaxis, :] ** 2).sum(axis=1) / sumw - mean ** 2,
                              np.nan)
        else:
            other_centers = h2.axes["x"].centers
            bin_centers = h2.axes["y"].centers
            edges = h2.axes["y"].edges
            sumw = w.sum(axis=0)
            mean = np.where(sumw > 0,
                            (w * other_centers[:, np.newaxis]).sum(axis=0) / sumw,
                            np.nan)
            spread = np.where(sumw > 0,
                              (w * other_centers[:, np.newaxis] ** 2).sum(axis=0) / sumw - mean ** 2,
                              np.nan)

    spread = np.maximum(spread, 0)
    n_eff = np.where(sumw > 0, sumw, 1)
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

    if lumi is not None:
        hep.cms.label("Preliminary", data=False, lumi=lumi, year="2024", ax=ax)
    else:
        hep.cms.label("Preliminary", data=False, ax=ax)

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
    samples = list(h.axes["dataset"])

    projected = {}
    for s in samples:
        h2 = h[{"dataset": s}]
        h1 = h2.project(proj_axis)
        ds_axis = hist.axis.StrCategory([s], name="dataset", growth=True)
        combined = hist.Hist(ds_axis, *h1.axes, storage=hist.storage.Weight())
        combined.view()[0, :] = h1.view()
        if s not in projected:
            projected[s] = combined
        else:
            projected[s] += combined

    merged_h = None
    for s, hh in projected.items():
        if merged_h is None:
            merged_h = hh
        else:
            merged_h += hh

    hist_cfg = {
        "label": cfg.get("label_x", h.axes[proj_axis].label),
    }
    plot_histogram(merged_h, hist_cfg, sample_defs, output_dir, name,
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

    if lumi is not None:
        hep.cms.label("Preliminary", data=False, lumi=lumi, year="2024", ax=ax)
    else:
        hep.cms.label("Preliminary", data=False, ax=ax)

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

    if lumi is not None:
        hep.cms.label("Preliminary", data=False, lumi=lumi, year="2024", ax=ax)
    else:
        hep.cms.label("Preliminary", data=False, ax=ax)

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
    Per-sample histograms are merged before plotting.
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
