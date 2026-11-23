"""Overlay gen-matched signal clusters against all background clusters.

The two populations live in different output directories: the signal's
gen-matched histograms come from a truth-level run, the background's inclusive
ones from a reco run (the background has no truth branches, so *all* its
clusters are the fake/pile-up reference).  Because both configs fill the same
variables with the same binning, the two can be drawn on one canvas without
refilling anything.

    python scripts/compare_sig_bkg.py <sig_out> <bkg_out> [<bkg_out> ...] <dest> \\
        --sig-cut matched --bkg-cut all

Region histograms (``<region>/<cut>/<variable>``) are paired by cut:
``csc/matched/csc_cluster_size`` in the signal run with
``csc/all/csc_cluster_size`` in the background runs.  Older flat names are
paired by suffix instead (``--suffix _matched``).  Several background
directories are read as one (each sample is one population, e.g. the
configs_mds_src_* runs), so there is no need to collect them first.

Each figure is unit-area normalized, so it compares shapes rather than yields,
and follows the same CMS style, sample labels/colors, and png+pdf output as
``suep-plot``.  Data samples are drawn as points, everything else as a fill
(background) or a step curve (signal, honouring the sample's ``linestyle``),
with the MC-stat uncertainty as a band (hatched on fills, shaded on curves);
``-c`` re-reads label/color/linestyle from a config so styling can be iterated
without reprocessing.

Each plot's separation -- the total-variation distance from the first
background, 0 = identical, 1 = no overlap -- is reported in ``separation.txt``
and orders the gallery, but is kept out of the legend.
"""
import argparse
import os
import sys

import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_style as style  # noqa: E402


def same_edges(a, b) -> bool:
    """Whether two 1D histograms have the same bin edges."""
    ea, eb = a.axes[0].edges, b.axes[0].edges
    return ea.shape == eb.shape and bool(np.allclose(ea, eb))


def collect(entries, base_or_key, hist_cfg, dropped=None, sample_of=None):
    """Unit-normalized 1D slices for [(sample, result_dict)] on a common axis.

    A sample that does not have this histogram, or has it empty, is left out
    rather than killing the whole comparison: the background sources are not
    all the same ntuple.  Cosmics MDSnano carries muon rechits only, so it has
    no ``dr_muon`` plot to contribute -- but signal-vs-DY-vs-ZeroBias for that
    variable is still worth drawing.  Pass *dropped* (a dict) to collect what
    was left out, so the caller can report it once instead of per plot.

    *sample_of* maps an entry's name to its dataset when the two differ (one
    sample drawn under several cuts).

    Returns [(sample, Hist)], or None when nothing survives or the survivors
    disagree on binning.
    """
    out, first = [], None
    for sample, data, key in entries:
        dataset = (sample_of or {}).get(sample, sample)
        h = data["histograms"].get(key)
        sh = None if h is None else style.slice_1d(h, dataset, hist_cfg)
        sh = None if sh is None else style.normalized(sh)
        if sh is None:
            if dropped is not None:
                dropped.setdefault(sample, set()).add(base_or_key)
            continue
        if first is None:
            first = sh
        elif not same_edges(first, sh):
            print(f"  !! axis mismatch, skipping {base_or_key}")
            return None
        out.append((sample, sh))
    return out or None


def sig_tag(suffix: str) -> str:
    """How to describe the signal population in legends and headings."""
    if suffix == "_matched":
        return "gen-matched"
    return suffix.lstrip("_").replace("_", " ") or "all clusters"


def bkg_label(suffix: str) -> str:
    """How to describe the background population in legends and headings."""
    return f"{suffix.lstrip('_').replace('_', ' ')} clusters" if suffix \
        else "all clusters"


# Data points sit on top of up to three translucent MC fills, so they are drawn
# larger than the mplhep default: at the default size they read as part of the
# fill edge rather than as their own population.
DATA_MARKERSIZE = 9
DATA_ELINEWIDTH = 2.0


def _legend(sample_defs, sample: str, tag: str) -> str:
    """``<sample>: <population>``, or just the sample when it *is* the population.

    ``--tag ''`` is how the configs_mds_src_* sets are drawn: there each sample
    is one cluster population already (its samples.yaml says which), so a
    trailing tag would only repeat it.
    """
    label = style.sample_label(sample_defs, sample)
    return f"{label}: {tag}" if tag else label


def _headroom(ax, n_entries: int) -> None:
    """Grow the y-axis so the legend clears the tallest curve.

    The legend needs roughly one line height per entry plus a margin; on a log
    axis that is a factor per decade, on a linear one a fraction of the range.
    This is only the first guess -- ``_clear_legend`` measures the drawn legend
    and grows the axis further if it still overlaps.
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


def _clear_legend(fig, ax, legend, y_max: float, tries: int = 6) -> None:
    """Grow the y-axis until the drawn legend sits above the tallest bin.

    ``_headroom``'s per-entry estimate underestimates a legend of eight long
    sample names, so the guess is checked against the real rendered box: both
    the legend's bottom edge and the data's peak are converted to axes
    fractions, and the axis grows until they no longer overlap.
    """
    for _ in range(tries):
        fig.canvas.draw()
        to_axes = ax.transAxes.inverted()
        legend_bottom = to_axes.transform(legend.get_window_extent().p0)[1]
        peak = to_axes.transform(ax.transData.transform((ax.get_xlim()[0], y_max)))[1]
        if peak < legend_bottom - 0.02:
            return
        _headroom(ax, 2)


def _restyle(sample_defs: dict, hist_defs: dict, config_dirs) -> None:
    """Refresh plot-time sample and histogram keys from configs.

    The same override suep-plot -c applies (suep_plot.results.restyle), so a
    label, colour, marker, rebin or log-axis change is picked up here exactly
    as it is there, without reprocessing.
    """
    from suep_plot.results import restyle

    for cdir in config_dirs or []:
        restyle(hist_defs, sample_defs, cdir)


SYS = "<SYS>"


def _cut_system(cut: str, template: str):
    """The system a cut name fills a ``<SYS>`` template with, or None.

    ``csc_src`` against ``<SYS>_src`` gives ``csc``; a template without the
    placeholder matches only itself (and gives "").
    """
    if SYS not in template:
        return "" if cut == template else None
    head, tail = template.split(SYS, 1)
    if not (cut.startswith(head) and cut.endswith(tail)):
        return None
    sys_name = cut[len(head): len(cut) - len(tail)]
    return sys_name if sys_name.isidentifier() else None


def _pairs(sig_keys, args):
    """(signal histogram, {background cut: histogram}, figure name) to compare.

    With region histograms, the signal's ``<region>/<sig-cut>/<var>`` is paired
    with ``<region>/<bkg-cut>/<var>`` for every ``--bkg-cut``; a ``<SYS>`` in
    the cuts is the system (csc, dt), so ``--sig-cut '<SYS>_src' --bkg-cut
    '<SYS>_iso'`` pairs csc_src with csc_iso and dt_src with dt_iso.
    """
    if args.sig_cut:
        out = []
        for key in sig_keys:
            parts = key.split("/")
            if len(parts) != 3:
                continue
            region, cut, leaf = parts
            sys_name = _cut_system(cut, args.sig_cut)
            if sys_name is None:
                continue
            cuts = list(args.bkg_cut)
            if getattr(args, "data", None):
                cuts.append(args.data_cut)
            bkg = {b: f"{region}/{b.replace(SYS, sys_name)}/{leaf}" for b in cuts}
            out.append((key, bkg, f"{region}/{leaf}"))
        return out
    keys = [k for k in sig_keys if k.endswith(args.suffix)] if args.suffix else sig_keys
    out = []
    for key in keys:
        base = key[: len(key) - len(args.suffix)] if args.suffix else key
        out.append((key, {"": base + args.bkg_suffix}, base))
    return out


def cut_label(cut: str) -> str:
    """``<SYS>_in_jet`` -> ``in jet``: the cut as a legend phrase."""
    return cut.replace(SYS, "").strip("_").replace("_", " ") or "all clusters"


def summed_background(bkg_all, names, key, hist_cfg, sample_defs, dropped=None):
    """The xs-weighted sum of every background sample, unit-normalized.

    Each sample enters with weight xs / sumw (its cross section over its summed
    generator weight), the --lumi scaling of suep-plot up to the luminosity,
    which drops out of a unit-area curve.  So the HT / jet-multiplicity bins
    of one process and the processes themselves add up as they would in data.
    A sample without xs or sumw stops the comparison: leaving it out would
    silently change the mixture.
    """
    total = None
    for name in names:
        data = bkg_all[name]
        h = data["histograms"].get(key)
        sh = None if h is None else style.slice_1d(h, name, hist_cfg)
        if sh is None:
            if dropped is not None:
                dropped.setdefault(name, set()).add(key)
            continue
        xs = sample_defs.get(name, {}).get("xs")
        sumw = data.get("sumw", {}).get(name)
        if xs is None or not sumw:
            raise SystemExit(f"--bkg-sum: background sample '{name}' has no "
                             f"{'xs' if xs is None else 'sumw'}")
        scaled = sh * (float(xs) / sumw)
        if total is None:
            total = scaled
        elif not same_edges(total, scaled):
            print(f"  !! axis mismatch, skipping {key}")
            return None
        else:
            total = total + scaled
    return None if total is None else style.normalized(total)


def draw(sig_slices, bkg_slices, sample_defs, base, hist_cfg, dest, tag, bkg_tag):
    """One canvas per variable: every signal (line) over every background (fill).

    Returns {background sample: mean separation of the signals from it}.  Every
    background is scored, because which one is the meaningful reference depends
    on the variable: a DY-derived cut can be an artifact of DY's dimuon
    selection while the same variable separates cleanly from minimum bias.
    """
    fig, ax, _rax = style.figure()
    colors = style.palette()

    def is_data(sample):
        return bool(sample_defs.get(sample, {}).get("is_data", False))

    for i, (sample, sh) in enumerate(bkg_slices):
        cfg = sample_defs.get(sample, {})
        color = cfg.get("color", colors[i % len(colors)])
        label = _legend(sample_defs, sample, bkg_tag)
        if is_data(sample):
            # data reads as points, so it never blends into an MC fill behind it
            hep.histplot(sh, ax=ax, histtype="errorbar", color=color, label=label,
                         markersize=DATA_MARKERSIZE, elinewidth=DATA_ELINEWIDTH,
                         yerr=np.sqrt(sh.variances()))
        else:
            hep.histplot(sh, ax=ax, histtype="fill", color=color, alpha=0.25,
                         edgecolor=color, linewidth=2, label=label)
            # hatched: a translucent band would vanish into the translucent fill
            style.draw_uncertainty_band(ax, sh, color=color, hatch="////")

    seps = {b: [] for b, _ in bkg_slices}
    for i, (sample, sh) in enumerate(sig_slices):
        for bkg, bsh in bkg_slices:
            seps[bkg].append(0.5 * np.abs(sh.values() - bsh.values()).sum())
        cfg = sample_defs.get(sample, {})
        color = cfg.get("color", colors[(i + len(bkg_slices)) % len(colors)])
        label = _legend(sample_defs, sample, tag)
        if is_data(sample):
            hep.histplot(sh, ax=ax, histtype="errorbar", color=color, label=label,
                         markersize=DATA_MARKERSIZE, elinewidth=DATA_ELINEWIDTH,
                         yerr=np.sqrt(sh.variances()))
        else:
            hep.histplot(sh, ax=ax, histtype="step", linewidth=2, color=color,
                         linestyle=cfg.get("linestyle", "-"), label=label, yerr=False)
            style.draw_uncertainty_band(ax, sh, color=color)

    ax.set_xlabel(style.x_label(hist_cfg, base))
    ax.set_ylabel("Normalized")
    if hist_cfg.get("log_x"):
        ax.set_xscale("log")

    allv = np.concatenate([sh.values()[sh.values() > 0]
                           for _, sh in bkg_slices + sig_slices])
    if allv.size and allv.min() / allv.max() < 1e-2:
        ax.set_yscale("log")
    n_entries = len(bkg_slices) + len(sig_slices)
    _headroom(ax, n_entries)
    # a signal grid plus two backgrounds is eight long names; shrink to fit
    fontsize = style.LEGEND_FONTSIZE - 4 if n_entries > 5 else style.LEGEND_FONTSIZE
    legend = ax.legend(fontsize=fontsize, loc="upper right")
    if allv.size:
        _clear_legend(fig, ax, legend, allv.max())

    has_data = any(sample_defs.get(s, {}).get("is_data", False)
                   for s, _ in bkg_slices + sig_slices)
    style.cms_label(ax, has_data=has_data)
    style.save(fig, dest, base)
    return {b: float(np.nanmean(v)) for b, v in seps.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sig_out", help="output dir of the truth-level (signal) run")
    p.add_argument("bkg_out", nargs="+",
                   help="output dir(s) of the background run(s), read as one")
    p.add_argument("dest", help="directory for the comparison figures")
    p.add_argument("--sig-cut", default=None,
                   help="region histograms: the signal's cut, e.g. matched -- pairs "
                        "<region>/<sig-cut>/<var> with <region>/<bkg-cut>/<var>")
    p.add_argument("--bkg-cut", nargs="+", default=["all"],
                   help="region histograms: the background's cut(s); several draw "
                        "one background curve per cut. '<SYS>' in --sig-cut and "
                        "here stands for the system, e.g. --sig-cut '<SYS>_src' "
                        "--bkg-cut '<SYS>_in_jet' '<SYS>_on_muon' '<SYS>_iso'")
    p.add_argument("--bkg-sum", action="store_true",
                   help="draw the background as one curve per cut: every background "
                        "sample summed with weight xs/sumw (the --lumi mixture of "
                        "suep-plot), then unit-area normalized")
    p.add_argument("--bkg-colors", nargs="+", default=None,
                   help="colours of the summed background curves, one per --bkg-cut "
                        "(default: the colour cycle)")
    p.add_argument("--data", nargs="+", default=None, metavar="DATA_OUT",
                   help="output dir(s) of data runs drawn as points under their own "
                        "cut (--data-cut), never part of the --bkg-sum: e.g. the "
                        "ZeroBias clusters next to the summed MC")
    p.add_argument("--data-cut", default="<SYS>_src",
                   help="region histograms: the cut of the --data runs")
    p.add_argument("--suffix", default="_matched",
                   help="flat histogram names: signal suffix to pair with the "
                        "background histogram (ignored with --sig-cut)")
    p.add_argument("--bkg-suffix", default="",
                   help="background histogram suffix (default: the inclusive "
                        "histogram). Set it to put the same cluster selection on "
                        "both sides, e.g. --suffix _matched_iso_jet "
                        "--bkg-suffix _iso_jet")
    p.add_argument("--tag", default=None,
                   help="what to call the signal population in legends, instead "
                        "of the name derived from --suffix. Pass '' to drop it, "
                        "for pickles whose sample labels already say which "
                        "population they hold (the configs_mds_src_* sets)")
    p.add_argument("--bkg-tag", default=None,
                   help="the same for the background population")
    p.add_argument("--sig-samples", nargs="*", default=None)
    p.add_argument("--bkg-samples", nargs="*", default=None)
    p.add_argument("--rank-against", default=None, metavar="SAMPLE",
                   help="background whose separation orders the gallery "
                        "(default: the first one flagged is_data, else the first)")
    p.add_argument("-c", "--config-dir", nargs="*", default=None,
                   help="config dir(s) whose samples.yaml re-supplies label / color / "
                        "linestyle at plot time, overriding what the pickles stored")
    args = p.parse_args()

    sig_all = style.load_samples(args.sig_out, args.sig_samples)
    if not sig_all:
        wanted = f" matching {args.sig_samples}" if args.sig_samples else ""
        raise SystemExit(f"no signal pickles{wanted} found in {args.sig_out}")
    bkg_all = {}
    for bkg_dir in args.bkg_out:
        for name, data in style.load_samples(bkg_dir, args.bkg_samples).items():
            if name in bkg_all:
                raise SystemExit(f"sample '{name}' is in more than one background "
                                 "directory; it would be counted twice")
            bkg_all[name] = data
    if not bkg_all:
        raise SystemExit(f"no background pickles found in {args.bkg_out}")

    # every signal sample and every background sample share one canvas per variable
    bkg_names = [n for n in bkg_all if n not in sig_all]
    if not bkg_names:
        raise SystemExit("background directory contains only signal samples")

    data_all = {}
    for data_dir in args.data or []:
        for name, data in style.load_samples(data_dir).items():
            if name in data_all or name in bkg_all or name in sig_all:
                raise SystemExit(f"--data sample '{name}' is also an input elsewhere")
            data_all[name] = data
    if args.data and not data_all:
        raise SystemExit(f"no data pickles found in {args.data}")

    sample_defs, hist_defs = {}, {}
    for data in list(sig_all.values()) + list(bkg_all.values()) + list(data_all.values()):
        sample_defs.update(data.get("samples", {}))
        hist_defs.update(data.get("hist_defs", {}))
    _restyle(sample_defs, hist_defs, args.config_dir)

    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    dropped: dict = {}
    sig_keys = sorted(next(iter(sig_all.values()))["histograms"])
    # An empty suffix pairs each signal histogram with the identically named
    # background one (for runs whose config defines no gen-matched histograms).
    pairs = _pairs(sig_keys, args)
    if not pairs:
        what = (f"a '{args.sig_cut}' cut" if args.sig_cut
                else f"histograms ending with '{args.suffix}'")
        raise SystemExit(
            f"no signal histogram with {what} in {args.sig_out} "
            f"({len(sig_keys)} histograms found).\n"
            "Region histograms are paired with --sig-cut <cut> [--bkg-cut <cut>]; "
            "flat names with --suffix ('' compares identically named histograms).")

    bkg_cuts = args.bkg_cut if args.sig_cut else [""]
    several = len(bkg_cuts) > 1
    if args.sig_cut:
        default_tag = cut_label(args.sig_cut)
        # with several cuts (or a summed background) each curve's own label
        # names its cut, so a common tag would only repeat it
        default_bkg = ("" if several or args.bkg_sum or bkg_cuts == ["all"]
                       else cut_label(bkg_cuts[0]))
    else:
        default_tag, default_bkg = sig_tag(args.suffix), bkg_label(args.bkg_suffix)
    tag = default_tag if args.tag is None else args.tag
    bkg_tag = default_bkg if args.bkg_tag is None else args.bkg_tag

    # The background curves: one per sample (and cut, with several cuts), or
    # with --bkg-sum one per cut.  Each gets a sample_defs entry, so draw()
    # finds its label and colour the same way as for a plain sample.
    colors = args.bkg_colors or style.palette()
    curves = []   # [(curve name, background cut, sample or None for the sum)]
    for i, cut in enumerate(bkg_cuts):
        if args.bkg_sum:
            name = f"sum:{cut_label(cut).replace(' ', '_')}"
            sample_defs[name] = {"label": f"SM background, {cut_label(cut)}",
                                 "color": colors[i % len(colors)]}
            curves.append((name, cut, None))
            continue
        for sample in bkg_names:
            name = f"{sample}:{cut_label(cut).replace(' ', '_')}" if several else sample
            if several:
                cfg = dict(sample_defs.get(sample, {}))
                cfg["label"] = f"{style.sample_label(sample_defs, sample)}, {cut_label(cut)}"
                sample_defs[name] = cfg
            curves.append((name, cut, sample))
    curve_names = [c[0] for c in curves] + list(data_all)

    seps = {}   # {variable: {background curve: separation}}
    for key, bkg_keys, base in pairs:
        # the background config carries the axis label
        hist_cfg = hist_defs.get(next(iter(bkg_keys.values())), hist_defs.get(key, {}))
        sig_entries = [(n, d, key) for n, d in sig_all.items()]
        sig_slices = collect(sig_entries, base, hist_cfg, dropped)
        if args.bkg_sum:
            bkg_slices = []
            for name, cut, _ in curves:
                sh = summed_background(bkg_all, bkg_names, bkg_keys[cut], hist_cfg,
                                       sample_defs, dropped)
                if sh is not None:
                    bkg_slices.append((name, sh))
            if bkg_slices and not all(same_edges(bkg_slices[0][1], s) for _, s in bkg_slices):
                print(f"  !! axis mismatch between background cuts, skipping {base}")
                continue
        else:
            bkg_entries = [(name, bkg_all[sample], bkg_keys[cut])
                           for name, cut, sample in curves]
            bkg_slices = collect(bkg_entries, base, hist_cfg, dropped,
                                 sample_of={name: sample for name, _, sample in curves})
        if data_all and bkg_slices:
            data_entries = [(n, d, bkg_keys[args.data_cut]) for n, d in data_all.items()]
            bkg_slices += collect(data_entries, base, hist_cfg, dropped) or []
        if not sig_slices or not bkg_slices:
            continue
        # collect() compares within each side only; the separation is taken
        # bin by bin across the two, so they need the same axis as well.
        if not all(same_edges(sig_slices[0][1], b) for _, b in bkg_slices):
            print(f"  !! signal and background binning differ, skipping {base}")
            continue
        by_bkg = draw(sig_slices, bkg_slices, sample_defs, base, hist_cfg, dest,
                      tag, bkg_tag)
        if by_bkg:
            seps[base] = by_bkg

    if dropped:
        print("Samples left out of some comparisons (no such histogram, or empty):")
        for sample, variables in sorted(dropped.items()):
            print(f"  {sample}: {len(variables)} variable(s)")

    if not seps:
        raise SystemExit("no variable could be drawn")

    ref = args.rank_against or next(
        (n for n in curve_names if sample_defs.get(n, {}).get("is_data")), curve_names[0])
    if ref not in curve_names:
        raise SystemExit(f"--rank-against {ref!r} is not one of {curve_names}")
    ranked = sorted(((base, by_bkg[ref]) for base, by_bkg in seps.items()
                     if ref in by_bkg), key=lambda kv: -kv[1])

    # plain sample names here: labels may carry mathtext, which only the
    # figures render -- the txt summary and the gallery header would show it raw
    sig_list, bkg_list = ", ".join(sig_all), ", ".join(curve_names)
    width = max(len(n) for n in curve_names) + 2
    with open(os.path.join(dest, "separation.txt"), "w") as fh:
        fh.write(f"# mean separation of [{sig_list}] from each background, "
                 f"1 = no overlap; ordered by {ref}\n")
        fh.write("#" + "".join(f"{n:>{width}}" for n in curve_names) + "  variable\n")
        for base, _ in ranked:
            row = "".join(f"{seps[base].get(n, float('nan')):>{width}.4f}"
                          for n in curve_names)
            fh.write(f" {row}  {base}\n")
    gallery = style.write_gallery(
        dest, f"Signal {tag or 'clusters'} ({sig_list}) vs "
              f"{bkg_tag or 'clusters'} ({bkg_list})",
        ("Background samples summed with weight xs/sumw. " if args.bkg_sum else "")
        + f"Unit-area normalized, sorted by separation from {ref}.", ranked, "separation")
    print(f"{len(ranked)} plots -> {gallery}")
    print(f"{'':7}" + "".join(f"{n:>{width}}" for n in curve_names))
    for base, _ in ranked[:10]:
        row = "".join(f"{seps[base].get(n, float('nan')):>{width}.3f}" for n in curve_names)
        print(f"{row}   {base}")


if __name__ == "__main__":
    main()
