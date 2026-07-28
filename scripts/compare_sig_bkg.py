"""Overlay gen-matched signal clusters against all background clusters.

The two populations live in different output directories: the signal's
``<var>_matched`` histograms come from a configs_mds_llp run (truth needed),
the background's inclusive ``<var>`` from a configs_mds run (the background has
no truth branches, so *all* its clusters are the fake/pile-up reference).
Because both configs define the inclusive histograms identically, the axes
match and the two can be drawn on one canvas without refilling anything.

    python scripts/compare_sig_bkg.py <sig_out> <bkg_out> <dest> [--suffix _matched]

Each figure is unit-area normalized, so it compares shapes rather than yields.
The separation quoted per plot is the total-variation distance between the two
shapes: 0 = identical, 1 = no overlap.
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_dir(outdir, samples=None):
    """Sum the pickles of an output directory into {sample: {hist: Hist}}."""
    out = {}
    for path in sorted(glob.glob(f"{outdir}/*.pkl")):
        sample = os.path.basename(path).split(".")[0]
        if samples and sample not in samples:
            continue
        hists = pickle.load(open(path, "rb"))["histograms"]
        if sample not in out:
            out[sample] = {k: v.copy() for k, v in hists.items()}
        else:
            for k, v in hists.items():
                out[sample][k] += v
    return out


def shape(hist):
    """Unit-normalized 1D values, or None if the histogram isn't usable."""
    values = hist.values()
    if values.ndim == 2:                     # drop the 'dataset' axis
        values = values.sum(axis=0)
    if values.ndim != 1:
        return None
    total = values.sum()
    return values / total if total > 0 else None


SIG_COLORS = ["tab:red", "tab:orange", "tab:green", "tab:purple"]


def draw(sig_entries, bkg_entries, base, dest):
    """One canvas per variable: every signal (line) over every background (filled).

    ``sig_entries``/``bkg_entries`` are [(sample_name, Hist)].  Returns the mean
    separation of the signals from the first background, or None if unusable.
    """
    bkg_shapes, sig_shapes = [], []
    edges = None
    for name, hist in bkg_entries + sig_entries:
        s = shape(hist)
        if s is None:
            return None
        if edges is None:
            edges = hist.axes[-1].edges
        elif edges.shape != hist.axes[-1].edges.shape or not np.allclose(
                edges, hist.axes[-1].edges):
            print(f"  !! axis mismatch, skipping {base}")
            return None
        (bkg_shapes if (name, hist) in bkg_entries else sig_shapes).append((name, s))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for name, s in bkg_shapes:
        ax.stairs(s, edges, color="tab:blue", lw=1.4, fill=True, alpha=0.25,
                  label=f"{name}: all clusters")
        ax.stairs(s, edges, color="tab:blue", lw=1.4)
    seps = []
    for i, (name, s) in enumerate(sig_shapes):
        sep = 0.5 * np.abs(s - bkg_shapes[0][1]).sum() if bkg_shapes else np.nan
        seps.append(sep)
        ax.stairs(s, edges, color=SIG_COLORS[i % len(SIG_COLORS)], lw=1.8,
                  label=f"{name}: gen-matched  (sep {sep:.2f})")

    ax.set_xlabel(base)
    ax.set_ylabel("unit-normalized")
    ax.set_title(base, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    allv = np.concatenate([s[s > 0] for _, s in bkg_shapes + sig_shapes])
    if allv.size and allv.min() / allv.max() < 1e-2:
        ax.set_yscale("log")
    fig.savefig(os.path.join(dest, f"{base}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
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
    args = p.parse_args()

    sig_all = load_dir(args.sig_out, args.sig_samples)
    bkg_all = load_dir(args.bkg_out, args.bkg_samples)
    if not bkg_all:
        raise SystemExit(f"no background pickles found in {args.bkg_out}")

    # every signal sample and every background sample share one canvas per variable
    bkg_names = [n for n in bkg_all if n not in sig_all]
    if not bkg_names:
        raise SystemExit("background directory contains only signal samples")
    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    seps = {}
    for key in sorted(next(iter(sig_all.values()))):
        if not key.endswith(args.suffix):
            continue
        base = key[: -len(args.suffix)]
        sig_entries = [(n, h[key]) for n, h in sig_all.items() if key in h]
        bkg_entries = [(n, bkg_all[n][base]) for n in bkg_names if base in bkg_all[n]]
        if not sig_entries or not bkg_entries:
            continue
        sep = draw(sig_entries, bkg_entries, base, dest)
        if sep is not None:
            seps[base] = sep

    ranked = sorted(seps.items(), key=lambda kv: -kv[1])
    sig_list, bkg_list = ", ".join(sig_all), ", ".join(bkg_names)
    with open(os.path.join(dest, "separation.txt"), "w") as fh:
        fh.write(f"# mean separation of [{sig_list}] from {bkg_names[0]}, 1 = no overlap\n")
        for base, sep in ranked:
            fh.write(f"{sep:7.4f}  {base}\n")
    with open(os.path.join(dest, "index.html"), "w") as fh:
        fh.write(f"<h2>Gen-matched signal ({sig_list}) vs all clusters ({bkg_list})</h2>\n"
                 "<p>Unit-area normalized, sorted by separation.</p>\n")
        for base, sep in ranked:
            fh.write(f'<div style="display:inline-block;margin:4px">'
                     f'<img src="{base}.png" width="430"><br>'
                     f'<small>{base} &mdash; {sep:.3f}</small></div>\n')
    print(f"{len(ranked)} plots -> {dest}/index.html")
    for base, sep in ranked[:10]:
        print(f"    {sep:6.3f}  {base}")


if __name__ == "__main__":
    main()
