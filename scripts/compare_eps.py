"""Overlay the same histogram from two DBSCAN-eps (dR) processings.

    python scripts/compare_eps.py <out_dr02> <out_dr04> <compare_dir> [label_a] [label_b]

Writes one figure per 1D histogram per sample: unit-normalized shapes for both
settings on the main panel, and their ratio (b/a) underneath, so a change in
shape is separated from a change in yield (quoted in the legend).
"""
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(outdir, sample):
    """Sum every part pickle of one sample into a single histogram dict."""
    total = None
    for path in sorted(glob.glob(f"{outdir}/{sample}*.pkl")):
        h = pickle.load(open(path, "rb"))["histograms"]
        if total is None:
            total = {k: v.copy() for k, v in h.items()}
        else:
            for k, v in h.items():
                total[k] += v
    return total


def project(hist):
    """Drop the 'dataset' axis; None for anything that isn't 1D after that."""
    values = hist.values()
    if values.ndim == 2:
        values = values.sum(axis=0)
    return values if values.ndim == 1 else None


def samples_in(outdir):
    names = {os.path.basename(p).split(".")[0] for p in glob.glob(f"{outdir}/*.pkl")}
    return sorted(names)


def compare(hist_a, hist_b, name, label_a, label_b, out_png):
    va, vb = project(hist_a), project(hist_b)
    if va is None or vb is None or va.shape != vb.shape:
        return False
    na, nb = va.sum(), vb.sum()
    if na <= 0 or nb <= 0:
        return False

    edges = hist_a.axes[-1].edges
    centers = hist_a.axes[-1].centers
    sa, sb = va / na, vb / nb

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(6.4, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    ax.stairs(sa, edges, color="tab:blue", lw=1.6, label=f"{label_a}  (N={na:.3g})")
    ax.stairs(sb, edges, color="tab:red", lw=1.6, label=f"{label_b}  (N={nb:.3g})")
    ax.set_ylabel("unit-normalized")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(name, fontsize=10)
    if sa.max() > 0 and sa[sa > 0].min() / sa.max() < 1e-2:
        ax.set_yscale("log")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sa > 0, sb / sa, np.nan)
    rax.axhline(1.0, color="grey", lw=0.8, ls="--")
    rax.plot(centers, ratio, ".", color="k", ms=4)
    rax.set_ylabel(f"{label_b}/{label_a}", fontsize=8)
    rax.set_xlabel(name)
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        hi = min(np.nanpercentile(finite, 95) * 1.3, 5.0)
        rax.set_ylim(0, max(hi, 1.5))

    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    # total-variation distance: 0 = identical shape, 1 = disjoint
    return 0.5 * np.abs(sa - sb).sum()


def main(argv):
    out_a, out_b, cmp_dir = argv[1], argv[2], argv[3]
    label_a = argv[4] if len(argv) > 4 else "dR=0.2"
    label_b = argv[5] if len(argv) > 5 else "dR=0.4"

    for sample in samples_in(out_a):
        a, b = load(out_a, sample), load(out_b, sample)
        if not a or not b:
            continue
        dest = os.path.join(cmp_dir, sample)
        os.makedirs(dest, exist_ok=True)
        scores = {}
        for name in sorted(a):
            if name not in b:
                continue
            d = compare(a[name], b[name], name, label_a, label_b,
                        os.path.join(dest, f"{name}.png"))
            if d is not False:
                scores[name] = d
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        with open(os.path.join(dest, "ranking.txt"), "w") as fh:
            fh.write(f"# shape distance ({label_a} vs {label_b}), 0 = identical\n")
            for name, d in ranked:
                fh.write(f"{d:7.4f}  {name}\n")
        with open(os.path.join(dest, "index.html"), "w") as fh:
            fh.write(f"<h2>{sample}: {label_a} vs {label_b}</h2>\n"
                     "<p>Sorted by shape change (most changed first).</p>\n")
            for name, d in ranked:
                fh.write(f'<div style="display:inline-block;margin:4px">'
                         f'<img src="{name}.png" width="430"><br>'
                         f'<small>{name} &mdash; shape dist {d:.3f}</small></div>\n')
        print(f"{sample}: {len(ranked)} comparisons -> {dest}/index.html")
        for name, d in ranked[:5]:
            print(f"    {d:6.3f}  {name}")


if __name__ == "__main__":
    main(sys.argv)
