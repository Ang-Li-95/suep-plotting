"""Overlay the same histograms from two processings (e.g. two DBSCAN eps).

    python scripts/compare_eps.py <run_a> <run_b> <compare_dir> [label_a] [label_b]

Writes one figure per 1D histogram per sample: unit-normalized shapes for both
runs on the main panel, and their ratio (b/a) underneath, so a change in shape
is separated from a change in yield (quoted in the legend).  Figures use the
same CMS style, labels, and png+pdf output as ``suep-plot``; ranking.txt and
the gallery order them by how much the shape changed.
"""
import argparse
import os
import sys

import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_style as style  # noqa: E402


def compare(h_a, h_b, sample, sample_defs, name, hist_cfg, label_a, label_b, dest):
    """Draw one variable for both processings; returns the shape distance."""
    sa = style.slice_1d(h_a, sample, hist_cfg)
    sb = style.slice_1d(h_b, sample, hist_cfg)
    if sa is None or sb is None or sa.values().shape != sb.values().shape:
        return False
    na, nb = sa.sum().value, sb.sum().value
    norm_a, norm_b = style.normalized(sa), style.normalized(sb)
    if norm_a is None or norm_b is None:
        return False

    fig, ax, rax = style.figure(ratio=True)

    colors = style.palette()
    for sh, label, n, color in ((norm_a, label_a, na, colors[0]),
                                (norm_b, label_b, nb, colors[1])):
        hep.histplot(sh, ax=ax, histtype="step", linewidth=2, color=color,
                     label=f"{label}  (N={n:.3g})", yerr=np.sqrt(sh.variances()))

    ax.set_ylabel("Normalized")
    ax.set_xlabel("")  # the shared x-axis is labelled on the ratio panel only
    ax.legend(fontsize=style.LEGEND_FONTSIZE, loc="best")
    if hist_cfg.get("log_x"):
        ax.set_xscale("log")

    va, vb = norm_a.values(), norm_b.values()
    if va.max() > 0 and va[va > 0].min() / va.max() < 1e-2:
        ax.set_yscale("log")

    ax.text(0.03, 0.97, style.sample_label(sample_defs, sample),
            transform=ax.transAxes, ha="left", va="top", fontsize=16)
    style.cms_label(ax)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(va > 0, vb / va, np.nan)
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    rax.errorbar(norm_a.axes[0].centers, ratio, fmt="o", color="black", markersize=5)
    rax.set_ylabel(f"{label_b}/{label_a}", fontsize=14)
    rax.set_xlabel(style.x_label(hist_cfg, name))
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        hi = min(np.nanpercentile(finite, 95) * 1.3, 5.0)
        rax.set_ylim(0, max(hi, 1.5))

    style.save(fig, dest, name)
    # total-variation distance: 0 = identical shape, 1 = disjoint
    return 0.5 * np.abs(va - vb).sum()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", help="first output directory (the reference)")
    p.add_argument("run_b", help="second output directory")
    p.add_argument("compare_dir", help="directory for the comparison figures")
    p.add_argument("label_a", nargs="?", default=None,
                   help="legend label of run_a (default: its directory name)")
    p.add_argument("label_b", nargs="?", default=None,
                   help="legend label of run_b (default: its directory name)")
    args = p.parse_args(argv)
    out_a, out_b, cmp_dir = args.run_a, args.run_b, args.compare_dir
    label_a = args.label_a or os.path.basename(os.path.normpath(out_a))
    label_b = args.label_b or os.path.basename(os.path.normpath(out_b))

    for sample in style.samples_in(out_a):
        a, b = style.load_sample(out_a, sample), style.load_sample(out_b, sample)
        if not a or not b:
            continue
        sample_defs = a.get("samples", {})
        hist_defs = a.get("hist_defs", {})
        dest = os.path.join(cmp_dir, sample)
        os.makedirs(dest, exist_ok=True)
        scores = {}
        for name in sorted(a["histograms"]):
            if name not in b["histograms"]:
                continue
            d = compare(a["histograms"][name], b["histograms"][name], sample,
                        sample_defs, name, hist_defs.get(name, {}),
                        label_a, label_b, dest)
            if d is not False:
                scores[name] = d
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        with open(os.path.join(dest, "ranking.txt"), "w") as fh:
            fh.write(f"# shape distance ({label_a} vs {label_b}), 0 = identical\n")
            for name, d in ranked:
                fh.write(f"{d:7.4f}  {name}\n")
        gallery = style.write_gallery(
            dest, f"{sample}: {label_a} vs {label_b}",
            "Unit-area normalized, sorted by shape change (most changed first).",
            ranked, "shape dist")
        print(f"{sample}: {len(ranked)} comparisons -> {gallery}")
        for name, d in ranked[:5]:
            print(f"    {d:6.3f}  {name}")


if __name__ == "__main__":
    main()
