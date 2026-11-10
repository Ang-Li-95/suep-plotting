"""Plots derived from filled histograms: ``derived_plots.yaml``.

Profiles and projections of a 2D histogram, efficiencies and ratios of two
histograms, and signal-vs-background overlays.  All of it is plot time: the
sources are histograms the run already filled.
"""

from __future__ import annotations

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .figures import (DEFAULT_FORMATS, MARKERSIZE, _finish, _series, _split_samples,
                      figure, is_2d_hist, palette, plot_histogram, prep_1d)


# ── the definitions ───────────────────────────────────────────────


def load_derived_plot_defs(path: str) -> dict:
    """A set's derived_plots.yaml, with region-aware entries expanded."""
    from .config import load_config_file
    return expand_region_derived(load_config_file(path), path)


def expand_region_derived(defs: dict, path="derived_plots.yaml") -> dict:
    """Region-aware derived plots -> one explicit entry per variable.

    ::

        matchfrac_csc:
          type: efficiency
          region: csc
          numerator: matched        # object cuts of that region
          denominator: all
          vars: [csc_cluster_size, csc_cluster_eta]

    becomes ``csc/matchfrac_csc/csc_cluster_size`` (numerator
    ``csc/matched/csc_cluster_size``, denominator
    ``csc/all/csc_cluster_size``) and the same for ``csc_cluster_eta``.
    Single-source types (profiles, projections) take ``cut:`` instead; a cut
    left out means the region's per-event histogram.
    """
    from .config import ConfigError
    from .histdefs import entry_leaf, region_hist_name

    out = {}
    for name, cfg in defs.items():
        if not isinstance(cfg, dict) or "region" not in cfg:
            out[name] = cfg
            continue
        where = f"{path}: derived plot '{name}'"
        cfg = dict(cfg)
        region, specs = cfg.pop("region"), cfg.pop("vars", None)
        entry = DERIVED_TYPES.get(cfg.get("type", ""))
        if entry is None:
            raise ConfigError(f"ERROR in {where}: unknown type '{cfg.get('type')}'; "
                              f"known: {', '.join(DERIVED_TYPES)}")
        if not specs:
            raise ConfigError(f"ERROR in {where}: a region-aware derived plot needs "
                              "'vars'")
        keys = entry[1]
        for spec in specs:
            leaf = entry_leaf(spec)
            one = dict(cfg)
            if keys == ("source",):
                one["source"] = region_hist_name(region, one.pop("cut", None), leaf)
            else:
                for key in keys:
                    one[key] = region_hist_name(region, cfg.get(key), leaf)
            out[f"{region}/{name}/{leaf}"] = one
    return out


# ── renderers ─────────────────────────────────────────────────────


def require_sources(histograms: dict, names, plot_name: str) -> bool:
    """Warn and return False if any source histogram is missing."""
    missing = [n for n in names if n not in histograms]
    if missing:
        print(f"  WARNING: source histogram(s) {missing} not found for "
              f"derived plot '{plot_name}'")
    return not missing


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
    if not require_sources(histograms, [cfg["source"]], name):
        return 0
    h = histograms[cfg["source"]]
    axis = "x" if cfg["type"] == "profile_x" else "y"

    fig, ax, _rax = figure()
    for _s, h2, label, color, ls, mk in _series(h, sample_defs):
        centers, mean, err, _edges = _profile(h2, axis)
        ax.errorbar(centers, mean, yerr=err, marker=mk, linestyle=ls,
                    label=label, color=color, markersize=MARKERSIZE, capsize=3,
                    linewidth=1.5)

    profiled = "y" if axis == "x" else "x"
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h.axes[axis].label),
                   ylabel=cfg.get("label_y", f"Mean {h.axes[profiled].label}"))


def _plot_projection(name, cfg, histograms, sample_defs, output_dir, *,
                     log_y=False, lumi=None, formats=DEFAULT_FORMATS,
                     normalize=False, ratio=True, **_):
    if not require_sources(histograms, [cfg["source"]], name):
        return 0
    h = histograms[cfg["source"]]
    proj_axis = "x" if cfg["type"] == "projection_x" else "y"

    # A projection is drawn as an ordinary 1D histogram, so the command-line
    # switches that apply to those (--normalize, --no-ratio) apply here too.
    plot_histogram(h.project("dataset", proj_axis),
                   {"label": cfg.get("label_x", h.axes[proj_axis].label)},
                   sample_defs, output_dir, name,
                   normalize=normalize, log_y=log_y, lumi=lumi, formats=formats,
                   ratio=ratio)
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
    if not require_sources(histograms, (num_name, den_name), name):
        return 0
    h_num, h_den = histograms[num_name], histograms[den_name]
    if is_2d_hist(h_num):
        return _plot_efficiency_2d(name, cfg, h_num, h_den, sample_defs,
                                   output_dir, lumi=lumi, formats=formats)

    fig, ax, _rax = figure()
    centers = h_num.axes["x"].centers
    top = 0.0
    for s, hn, label, color, ls, mk in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        passed = hn.view().value
        total = h_den[{"dataset": s}].view().value

        filled = total > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(filled, passed / total, 0.0)
        lo, hi = _clopper_pearson(passed, total)

        ax.errorbar(centers[filled], eff[filled],
                    yerr=[(eff - lo)[filled], (hi - eff)[filled]],
                    marker=mk, linestyle=ls, label=label, color=color,
                    markersize=MARKERSIZE, capsize=3, linewidth=1.5)
        if filled.any():
            top = max(top, float(hi[filled].max()))

    if cfg.get("spans"):
        _draw_spans(ax, cfg["spans"])
    # Efficiencies that live well below 1 get an axis matched to them, so the
    # uncertainty band is not squeezed into nothing by empty canvas.  The
    # headroom leaves the legend somewhere to sit.  `ylim` in the config pins
    # it explicitly.
    ax.set_ylim(*(cfg.get("ylim") or ((-0.02 * top, min(1.15, top * 1.45))
                                      if 0 < top < 0.7 else (-0.05, 1.15))))
    return _finish(fig, ax, output_dir, name, formats, lumi=lumi,
                   xlabel=cfg.get("label_x", h_num.axes["x"].label),
                   ylabel=cfg.get("label_y", "Efficiency"))


def _ratio_with_error(num, num_var, den, den_var):
    """num / den per bin with its uncertainty; NaN where the denominator is empty.

    The two histograms are taken as statistically independent, so the
    relative uncertainties add in quadrature:
    ``r * sqrt(var_n / n^2 + var_d / d^2)``, written so that an empty
    numerator still gets the denominator-scaled error of its own variance.
    For a numerator that is a subset of the denominator this overestimates --
    that case is what the ``efficiency`` type is for.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = den > 0
        ratio = np.where(ok, num / den, np.nan)
        err = np.where(ok, np.sqrt(num_var + ratio ** 2 * den_var) / den, np.nan)
    return ratio, err


def _plot_ratio(name, cfg, histograms, sample_defs, output_dir, *,
                lumi=None, formats=DEFAULT_FORMATS, **_):
    num_name, den_name = cfg["numerator"], cfg["denominator"]
    if not require_sources(histograms, (num_name, den_name), name):
        return 0
    h_num, h_den = histograms[num_name], histograms[den_name]

    fig, ax, _rax = figure()
    centers = h_num.axes["x"].centers
    for s, hn, label, color, ls, mk in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        nv, dv = hn.view(), h_den[{"dataset": s}].view()
        ratio, err = _ratio_with_error(nv.value, nv.variance, dv.value, dv.variance)
        ax.errorbar(centers, ratio, yerr=err, marker=mk, linestyle=ls,
                    label=label, color=color, markersize=MARKERSIZE, capsize=3,
                    linewidth=1.5)

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
    if not require_sources(histograms, (sig_name, bkg_name), name):
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

    fig, ax, _rax = figure()
    colors = palette()

    drawn = 0
    for i, (s, h) in enumerate(entries):
        sh = prep_1d(h[{"dataset": s}], cfg)
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
    for s, hn, label, _color, _ls, _mk in _series(h_num, sample_defs):
        if s not in h_den.axes["dataset"]:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            total = h_den[{"dataset": s}].view().value
            eff = np.where(total > 0, hn.view().value / total, np.nan)

        fig, ax, _rax = figure()
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


# ── dispatch ──────────────────────────────────────────────────────


# The derived-plot types, each with the config keys naming its source
# histograms.  One table, read both by plot_all (to decide which histograms a
# task needs) and by plot_derived (to dispatch it) -- the two used to
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


def plot_derived(name, cfg, histograms, sample_defs, output_dir, *,
                      log_y=False, lumi=None, formats=DEFAULT_FORMATS,
                      normalize=False, ratio=True) -> int:
    """Dispatch a single derived-plot definition to its renderer.

    *normalize* and *ratio* reach the renderers that draw ordinary 1D
    histograms (the projections); the others ignore them.
    """
    entry = DERIVED_TYPES.get(cfg.get("type", ""))
    if entry is None:
        print(f"  WARNING: unknown derived type '{cfg.get('type', '')}' for '{name}'")
        return 0
    return entry[0](name, cfg, histograms, sample_defs, output_dir,
                    log_y=log_y, lumi=lumi, formats=formats,
                    normalize=normalize, ratio=ratio)
