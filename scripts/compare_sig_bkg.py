"""Overlay gen-matched signal clusters against all background clusters.

The two populations live in different output directories: the signal's
``<var>_matched`` histograms come from a configs_mds_llp run (truth needed),
the background's inclusive ``<var>`` from a configs_mds run (the background has
no truth branches, so *all* its clusters are the fake/pile-up reference).
Because both configs define the inclusive histograms identically, the axes
match and the two can be drawn on one canvas without refilling anything.

    python scripts/compare_sig_bkg.py <sig_out> <bkg_out> <dest> [--suffix _matched]

Each figure is unit-area normalized, so it compares shapes rather than yields,
and follows the same CMS style, sample labels/colors, and png+pdf output as
``suep-plot``.  Data samples are drawn as points, everything else as a fill
(background) or a step curve (signal, honouring the sample's ``linestyle``);
``-c`` re-reads label/color/linestyle from a config so styling can be iterated
without reprocessing.

Each plot's separation -- the total-variation distance from the first
background, 0 = identical, 1 = no overlap -- is reported in ``separation.txt``
and orders the gallery, but is kept out of the legend.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_style as style  # noqa: E402


def collect(entries, base_or_key, hist_cfg):
    """Unit-normalized 1D slices for [(sample, result_dict)] on a common axis.

    Returns [(sample, Hist)] or None when a histogram is missing, empty, or
    binned differently from the first one.
    """
    out, edges = [], None
    for sample, data, key in entries:
        h = data["histograms"].get(key)
        if h is None:
            return None
        sh = style.slice_1d(h, sample, hist_cfg)
        if sh is None:
            return None
        sh = style.normalized(sh)
        if sh is None:
            return None
        if edges is None:
            edges = sh.axes[0].edges
        elif edges.shape != sh.axes[0].edges.shape or not np.allclose(
                edges, sh.axes[0].edges):
            print(f"  !! axis mismatch, skipping {base_or_key}")
            return None
        out.append((sample, sh))
    return out


def sig_tag(suffix: str) -> str:
    """How to describe the signal population in legends and headings."""
    if suffix == "_matched":
        return "gen-matched"
    return suffix.lstrip("_").replace("_", " ") or "all clusters"


def _headroom(ax, n_entries: int) -> None:
    """Grow the y-axis so the legend clears the tallest curve.

    The legend needs roughly one line height per entry plus a margin; on a log
    axis that is a factor per decade, on a linear one a fraction of the range.
    """
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        if lo <= 0 or hi <= lo:
            return
        span = np.log10(hi / lo)                       # decades currently drawn
        added = min(0.42 * n_entries, 0.9 * span)      # never more than ~half the axis
        ax.set_ylim(lo, hi * 10 ** added)
    else:
        ax.set_ylim(lo, lo + (hi - lo) * (1.0 + min(0.17 * n_entries, 1.0)))


def _restyle(sample_defs: dict, config_dirs) -> None:
    """Refresh plot-only sample keys (label, color, linestyle) from configs.

    The pickles carry the sample definitions as they were at fill time, so
    without this a legend fix would need the samples reprocessed.
    """
    from suep_plot.processor import load_samples

    for cdir in config_dirs or []:
        path = os.path.join(cdir, "samples.yaml")
        if not os.path.exists(path):
            raise SystemExit(f"--config-dir: no samples.yaml in {cdir}")
        for name, cfg in load_samples(path).items():
            if name in sample_defs:
                sample_defs[name].update(
                    {k: v for k, v in cfg.items()
                     if k in ("label", "color", "linestyle", "group", "is_data")})


def draw(sig_slices, bkg_slices, sample_defs, base, hist_cfg, dest, tag):
    """One canvas per variable: every signal (line) over every background (fill).

    Returns the mean separation of the signals from the first background.
    """
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    colors = style.palette()

    def is_data(sample):
        return bool(sample_defs.get(sample, {}).get("is_data", False))

    for i, (sample, sh) in enumerate(bkg_slices):
        cfg = sample_defs.get(sample, {})
        color = cfg.get("color", colors[i % len(colors)])
        label = f"{style.sample_label(sample_defs, sample)}: all clusters"
        if is_data(sample):
            # data reads as points, so it never blends into an MC fill behind it
            hep.histplot(sh, ax=ax, histtype="errorbar", color=color, label=label,
                         markersize=5, elinewidth=1.5, yerr=np.sqrt(sh.variances()))
        else:
            hep.histplot(sh, ax=ax, histtype="fill", color=color, alpha=0.25,
                         edgecolor=color, linewidth=2, label=label)

    seps = []
    ref = bkg_slices[0][1].values() if bkg_slices else None
    for i, (sample, sh) in enumerate(sig_slices):
        sep = 0.5 * np.abs(sh.values() - ref).sum() if ref is not None else np.nan
        seps.append(sep)
        cfg = sample_defs.get(sample, {})
        color = cfg.get("color", colors[(i + len(bkg_slices)) % len(colors)])
        label = f"{style.sample_label(sample_defs, sample)}: {tag}"
        if is_data(sample):
            hep.histplot(sh, ax=ax, histtype="errorbar", color=color, label=label,
                         markersize=5, elinewidth=1.5, yerr=np.sqrt(sh.variances()))
        else:
            hep.histplot(sh, ax=ax, histtype="step", linewidth=2, color=color,
                         linestyle=cfg.get("linestyle", "-"),
                         label=label, yerr=np.sqrt(sh.variances()))

    ax.set_xlabel(style.x_label(hist_cfg, base))
    ax.set_ylabel("Normalized")
    if hist_cfg.get("log_x"):
        ax.set_xscale("log")

    allv = np.concatenate([sh.values()[sh.values() > 0]
                           for _, sh in bkg_slices + sig_slices])
    if allv.size and allv.min() / allv.max() < 1e-2:
        ax.set_yscale("log")
    _headroom(ax, len(bkg_slices) + len(sig_slices))
    ax.legend(fontsize=style.LEGEND_FONTSIZE, loc="upper right")

    has_data = any(sample_defs.get(s, {}).get("is_data", False)
                   for s, _ in bkg_slices + sig_slices)
    style.cms_label(ax, has_data=has_data)
    style.save(fig, dest, base)
    return float(np.nanmean(seps)) if seps else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sig_out", help="output dir of the configs_mds_llp run")
    p.add_argument("bkg_out", help="output dir of the configs_mds run")
    p.add_argument("dest", help="directory for the comparison figures")
    p.add_argument("--suffix", default="_matched",
                   help="signal histogram suffix to pair with the inclusive one")
    p.add_argument("--sig-samples", nargs="*", default=None)
    p.add_argument("--bkg-samples", nargs="*", default=None)
    p.add_argument("-c", "--config-dir", nargs="*", default=None,
                   help="config dir(s) whose samples.yaml re-supplies label / color / "
                        "linestyle at plot time, overriding what the pickles stored")
    args = p.parse_args()

    sig_all = style.load_samples(args.sig_out, args.sig_samples)
    bkg_all = style.load_samples(args.bkg_out, args.bkg_samples)
    if not bkg_all:
        raise SystemExit(f"no background pickles found in {args.bkg_out}")

    # every signal sample and every background sample share one canvas per variable
    bkg_names = [n for n in bkg_all if n not in sig_all]
    if not bkg_names:
        raise SystemExit("background directory contains only signal samples")

    sample_defs, hist_defs = {}, {}
    for data in list(sig_all.values()) + list(bkg_all.values()):
        sample_defs.update(data.get("samples", {}))
        hist_defs.update(data.get("hist_defs", {}))
    _restyle(sample_defs, args.config_dir)

    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    sig_keys = sorted(next(iter(sig_all.values()))["histograms"])
    # An empty suffix pairs each signal histogram with the identically named
    # background one (for runs whose config defines no gen-matched histograms).
    keys = [k for k in sig_keys if k.endswith(args.suffix)] if args.suffix else sig_keys
    if not keys:
        raise SystemExit(
            f"no signal histogram ends with '{args.suffix}' in {args.sig_out} "
            f"({len(sig_keys)} histograms found).\n"
            "That run's config defines no gen-matched histograms -- either process it "
            "with a config that does (e.g. configs_mds_llp), or pass --suffix '' to "
            "compare the inclusive signal histograms with the background ones.")

    seps = {}
    for key in keys:
        base = key[: len(key) - len(args.suffix)] if args.suffix else key
        # the inclusive config carries the axis label; fall back to the matched one
        hist_cfg = hist_defs.get(base, hist_defs.get(key, {}))
        sig_entries = [(n, d, key) for n, d in sig_all.items()]
        bkg_entries = [(n, bkg_all[n], base) for n in bkg_names]
        sig_slices = collect(sig_entries, base, hist_cfg)
        bkg_slices = collect(bkg_entries, base, hist_cfg)
        if not sig_slices or not bkg_slices:
            continue
        sep = draw(sig_slices, bkg_slices, sample_defs, base, hist_cfg, dest,
                   sig_tag(args.suffix))
        if sep is not None:
            seps[base] = sep

    ranked = sorted(seps.items(), key=lambda kv: -kv[1])
    # plain sample names here: labels may carry mathtext, which only the
    # figures render -- the txt summary and the gallery header would show it raw
    sig_list, bkg_list = ", ".join(sig_all), ", ".join(bkg_names)
    with open(os.path.join(dest, "separation.txt"), "w") as fh:
        fh.write(f"# mean separation of [{sig_list}] from {bkg_names[0]}, 1 = no overlap\n")
        for base, sep in ranked:
            fh.write(f"{sep:7.4f}  {base}\n")
    gallery = style.write_gallery(
        dest, f"Signal {sig_tag(args.suffix)} ({sig_list}) vs all clusters ({bkg_list})",
        "Unit-area normalized, sorted by separation.", ranked, "separation")
    print(f"{len(ranked)} plots -> {gallery}")
    for base, sep in ranked[:10]:
        print(f"    {sep:6.3f}  {base}")


if __name__ == "__main__":
    main()
