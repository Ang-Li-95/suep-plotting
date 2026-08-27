"""r-z event display of the muon-system rechits, coloured by DBSCAN cluster.

One figure per event: every CSC/DT rechit is drawn at its global (|z|, r) over
the chamber layout (:func:`drawRZ`), hits belonging to the same DBSCAN cluster
share a colour, hits DBSCAN calls noise are light grey.  The clustering is the
one the analysis uses (``custom/columns.py``: dR metric in eta-phi, same eps /
min_samples), so the shapes on the display are the objects the cluster
histograms are filled from -- pass ``-c <config set>`` to pick up that set's
``columns.yaml`` parameters.  RPC is not drawn.

    conda activate mds
    export X509_USER_PROXY=$HOME/private/.proxy
    python scripts/event_display.py -d suep_mDark2_temp1 -n 5

The chamber layout covers z > 0 only, so the display is folded to |z|; a
cluster in the -z endcap therefore lands on the same picture as one in +z (the
colours still separate them).

Only events with at least one cluster are drawn.  ``--matched-only`` restricts
to events with a truth-matched cluster and ``--zoom`` crops the axes to the
clustered hits (the shape of a single shower is a few tens of cm across, so the
full-detector view is mostly empty space).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
import yaml
from matplotlib import patches
from sklearn.cluster import DBSCAN

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import custom.columns as columns  # noqa: E402
from suep_plot.processor import _resolve_files, load_columns_config  # noqa: E402

OUTDIR = Path("/groups/hephy/cms/ang.li/suep_plots/event_display")

SYSTEMS = {
    "csc": dict(coll="cscRechits", label="CSC"),
    "dt": dict(coll="dtRecHits", label="DT"),
}
HIT_BRANCHES = ("X", "Y", "Z", "Eta", "Phi", "Station")
TRUTH_BRANCHES = ("llpIdx", "llpSimX", "llpSimY", "llpSimZ")

# Marker = hit-level truth (the colour already carries the cluster), so a
# cluster's LLP hits and the pile-up/punch-through it swept up stay apart.
MATCHED_HIT = dict(marker="o", s=16, linewidths=0, alpha=0.85)
OTHER_HIT = dict(marker="x", s=16, linewidths=0.9, alpha=0.85)


def cluster_labels(eta, phi, eps, min_samples):
    """DBSCAN labels per rechit (-1 = noise), identical to ``_cluster_system``."""
    if len(eta) < min_samples:
        return np.full(len(eta), -1, dtype=np.int64)
    dphi = phi[:, None] - phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dmat = np.sqrt((eta[:, None] - eta[None, :]) ** 2 + dphi ** 2)
    return DBSCAN(eps=eps, min_samples=min_samples,
                  metric="precomputed").fit_predict(dmat)


def read_event(arrays, i, systems, settings):
    """Flat per-system numpy arrays for one event, with the DBSCAN labels."""
    out = {}
    for sys_ in systems:
        coll = SYSTEMS[sys_]["coll"]
        if f"{coll}_X" not in arrays.fields:
            continue
        hits = {b: ak.to_numpy(arrays[f"{coll}_{b}"][i]) for b in HIT_BRANCHES}
        if f"{coll}_llpIdx" in arrays.fields:
            hits.update({b: ak.to_numpy(arrays[f"{coll}_{b}"][i])
                         for b in TRUTH_BRANCHES})
        eps, min_samples = columns._dbscan_params(sys_, settings)
        hits["label"] = cluster_labels(hits["Eta"], hits["Phi"], eps, min_samples)
        out[sys_] = hits
    return out


def cluster_summary(hits, settings):
    """Per-cluster (label, size, matched llpIdx or -1) for one system."""
    summary = []
    for k in range(hits["label"].max() + 1):
        m = hits["label"] == k
        best, nbest = -1, 0
        if "llpIdx" in hits:
            idx = hits["llpIdx"][m]
            idx = idx[idx >= 0]
            if len(idx):
                uniq, cnt = np.unique(idx, return_counts=True)
                best, nbest = int(uniq[cnt.argmax()]), int(cnt.max())
        summary.append((k, int(m.sum()),
                        best if nbest >= settings["match_min_hits"] else -1))
    return summary


def drawRZ(ax, MB_xmin=0):
    """CSC/DT chamber layout in the (z, r) quarter view, z > 0.

    ``MB_xmin`` is where the barrel boxes start in z; 0 draws the DT wheels
    over their full |z| range.
    """
    ax.set_ylim(0, 750)
    ax.set_xlim(0, 1100)

    MB1 = patches.Rectangle((MB_xmin, 402), 661 - MB_xmin, 449 - 402, color='grey', alpha=0.3)
    MB2 = patches.Rectangle((MB_xmin, 490), 661 - MB_xmin, 533 - 490, color='grey', alpha=0.3)
    MB3 = patches.Rectangle((MB_xmin, 597), 661 - MB_xmin, 636 - 597, color='grey', alpha=0.3)
    MB4 = patches.Rectangle((MB_xmin, 700), 661 - MB_xmin, 738 - 700, color='grey', alpha=0.3)

    solenoid = patches.Rectangle((MB_xmin, 295), 661 - MB_xmin, 380 - 295, color='grey', alpha=0.3)

    ME22 = patches.Rectangle((791, 357), 850 - 791, 700 - 357, color='grey', alpha=0.3)
    ME32 = patches.Rectangle((911, 357), 970 - 911, 700 - 357, color='grey', alpha=0.3)
    ME42 = patches.Rectangle((1002, 357), 1063 - 1002, 700 - 357, color='grey', alpha=0.3)

    ME21 = patches.Rectangle((789, 139), 850 - 789, 345 - 139, color='grey', alpha=0.3)
    ME31 = patches.Rectangle((915, 160), 970 - 915, 345 - 160, color='grey', alpha=0.3)
    ME41 = patches.Rectangle((1002, 178), 1063 - 1002, 345 - 178, color='grey', alpha=0.3)

    ME11 = patches.Rectangle((580, 100), 632 - 580, 275 - 100, color='grey', alpha=0.3)
    ME12 = patches.Rectangle((668, 275), 724 - 668, 465 - 275, color='grey', alpha=0.3)
    ME13 = patches.Rectangle((686, 505), 724 - 686, 700 - 505, color='grey', alpha=0.3)

    for box in [ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, ME42,
                MB1, MB2, MB3, MB4, solenoid]:
        ax.add_patch(box)

    ax.text(300, 415, "MB1", fontsize=12)
    ax.text(300, 500, "MB2", fontsize=12)
    ax.text(300, 605, "MB3", fontsize=12)
    ax.text(300, 708, "MB4", fontsize=12)

    ax.text(665, 705, "ME1/3", fontsize=12)
    ax.text(780, 705, "ME2/2", fontsize=12)
    ax.text(900, 705, "ME3/2", fontsize=12)
    ax.text(990, 705, "ME4/2", fontsize=12)

    ax.text(300, 330, "Solenoid", fontsize=12)

    ax.text(615, 110, "ME1/1", rotation='vertical', fontsize=12)
    ax.text(695, 331, "ME1/2", rotation='vertical', fontsize=12)
    ax.text(830, 145, "ME2/1", rotation='vertical', fontsize=12)
    ax.text(955, 170, "ME3/1", rotation='vertical', fontsize=12)
    ax.text(1040, 190, "ME4/1", rotation='vertical', fontsize=12)
    return ax


def set_zoom(ax, x, y, floor):
    """Crop an axis to the points (x, y) with a margin, keeping a minimum span."""
    pad = 0.15 * max(np.ptp(x), np.ptp(y), floor)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)


def draw_event(event, meta, out_path, settings, zoom=False, with_etaphi=False,
               is_data=False):
    ncol = 2 if with_etaphi else 1
    fig, axes = plt.subplots(1, ncol, figsize=(9.0 * ncol, 7.5))
    ax = axes[0] if with_etaphi else axes
    # first: it fixes the axis limits, which the hits must not stretch
    drawRZ(ax)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    icolor = 0
    legend_lines = []
    drawn = {k: [] for k in ("z", "r", "eta", "phi")}  # clustered hits, for --zoom

    for sys_, hits in event.items():
        r = np.hypot(hits["X"], hits["Y"])
        z = np.abs(hits["Z"])  # the chamber layout covers z > 0 only
        # hit-level truth: does this rechit come from an LLP shower?  Without
        # truth branches (data / central background) every hit gets the plain
        # marker instead of being called "not from an LLP".
        from_llp = (hits["llpIdx"] >= 0 if "llpIdx" in hits
                    else np.ones(len(z), dtype=bool))
        noise = hits["label"] < 0
        if noise.any():
            ax.scatter(z[noise], r[noise], s=4, c="0.8", marker=".",
                       linewidths=0, zorder=1)
            if with_etaphi:
                axes[1].scatter(hits["Eta"][noise], hits["Phi"][noise], s=4,
                                c="0.8", marker=".", linewidths=0, zorder=1)
        for k, size, llp in cluster_summary(hits, settings):
            m = hits["label"] == k
            color = colors[icolor % len(colors)]
            icolor += 1
            # colour = cluster, marker = LLP-matched (o) or not (x); the system
            # is obvious from where the hits sit
            for sel, style in ((m & from_llp, MATCHED_HIT), (m & ~from_llp, OTHER_HIT)):
                if not sel.any():
                    continue
                ax.scatter(z[sel], r[sel], color=color, zorder=3, **style)
                if with_etaphi:
                    axes[1].scatter(hits["Eta"][sel], hits["Phi"][sel], color=color,
                                    zorder=3, **style)
            drawn["z"].append(z[m])
            drawn["r"].append(r[m])
            drawn["eta"].append(hits["Eta"][m])
            drawn["phi"].append(hits["Phi"][m])
            tag = f"LLP {llp}" if llp >= 0 else "unmatched"
            legend_lines.append(ax.scatter(
                [], [], s=30, color=color, marker="o",
                label=f"{SYSTEMS[sys_]['label']} #{k}: {size} hits, {tag}"))
            # the truth shower position of the matched LLP, for reference
            if llp >= 0 and "llpSimX" in hits:
                sel = m & (hits["llpIdx"] == llp)
                sx, sy, sz = (np.median(hits[b][sel])
                              for b in ("llpSimX", "llpSimY", "llpSimZ"))
                ax.scatter([abs(sz)], [np.hypot(sx, sy)], s=180, marker="*",
                           facecolor="none", edgecolor=color, linewidths=1.4,
                           zorder=4)

    ax.set_xlabel("$|z|$ [cm]")
    ax.set_ylabel("$r$ [cm]")
    if zoom and drawn["z"]:
        set_zoom(ax, np.concatenate(drawn["z"]), np.concatenate(drawn["r"]),
                 floor=50.0)
    has_truth = any("llpIdx" in hits for hits in event.values())
    if has_truth:
        legend_lines += [
            ax.scatter([], [], color="0.35", label="hit from an LLP", **MATCHED_HIT),
            ax.scatter([], [], color="0.35", label="hit not from an LLP", **OTHER_HIT),
        ]
    if legend_lines:
        # under the axes: the chamber layout leaves no free corner inside
        ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                  ncol=2, frameon=False,
                  title="open star: truth shower position" if has_truth else None,
                  title_fontsize=10)
    hep.cms.text("Preliminary" if is_data else "Simulation", ax=ax, fontsize=16)
    ax.set_title(meta, fontsize=12, loc="right", color="0.35")

    if with_etaphi:
        axes[1].set_xlabel(r"$\eta$")
        axes[1].set_ylabel(r"$\phi$")
        if zoom and drawn["eta"]:
            set_zoom(axes[1], np.concatenate(drawn["eta"]),
                     np.concatenate(drawn["phi"]), floor=0.3)
        else:
            axes[1].set_xlim(-2.6, 2.6)
            axes[1].set_ylim(-np.pi, np.pi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", default="suep_mDark2_temp1",
                   help="dataset name in datasets.yaml (default: suep_mDark2_temp1)")
    p.add_argument("-f", "--files", nargs="+",
                   help="explicit ROOT file(s), overrides --dataset")
    p.add_argument("--nfiles", type=int, default=1,
                   help="how many files of the dataset to scan (default: 1)")
    p.add_argument("-n", "--nevents", type=int, default=5,
                   help="how many event displays to draw (default: 5)")
    p.add_argument("--entries", type=int, nargs="+",
                   help="draw these entry numbers of the first file instead of "
                        "scanning for events with clusters")
    p.add_argument("--systems", default="csc,dt",
                   help="comma-separated subset of csc,dt (default: both)")
    p.add_argument("--min-size", type=int, default=0,
                   help="only draw events with a cluster of at least this size")
    p.add_argument("--matched-only", action="store_true",
                   help="only draw events with a truth-matched cluster")
    p.add_argument("--zoom", action="store_true",
                   help="crop the axes to the clustered hits")
    p.add_argument("--with-etaphi", action="store_true",
                   help="add an eta-phi panel next to the r-z one")
    p.add_argument("-o", "--out", default=str(OUTDIR),
                   help=f"output directory (default: {OUTDIR})")
    p.add_argument("-c", "--config-dir", default=None,
                   help="config set whose columns.yaml sets the clustering "
                        "parameters (default: the module defaults)")
    args = p.parse_args()

    # Cluster exactly like the histograms of that config set were filled.
    settings = columns.configure(
        load_columns_config(Path(args.config_dir) / "columns.yaml")
        if args.config_dir else None)
    eps, min_samples = columns._dbscan_params("csc", settings)
    print(f"DBSCAN: eps={eps}, min_samples={min_samples} (CSC/DT), "
          f"match_min_hits={settings['match_min_hits']}")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        p.error(f"unknown system(s): {', '.join(sorted(unknown))}")

    if args.files:
        files, tag, is_data = list(args.files), "files", False
    else:
        with open(REPO / "datasets.yaml") as f:
            samples = yaml.safe_load(f)
        if args.dataset not in samples:
            p.error(f"unknown dataset '{args.dataset}'. Available: "
                    f"{', '.join(sorted(samples))}")
        files = _resolve_files(samples[args.dataset]["files"])[:args.nfiles]
        tag = args.dataset
        is_data = bool(samples[args.dataset].get("is_data", False))
    if not files:
        p.error("no input files resolved")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    branches = []
    for sys_ in systems:
        coll = SYSTEMS[sys_]["coll"]
        branches += [f"{coll}_{b}" for b in HIT_BRANCHES + TRUTH_BRANCHES]

    ndrawn = 0
    for path in files:
        tree = uproot.open(path)["Events"]
        present = [b for b in branches if b in tree]
        arrays = tree.arrays(present)
        entries = args.entries if args.entries is not None else range(len(arrays))
        for i in entries:
            if ndrawn >= args.nevents:
                break
            event = read_event(arrays, i, systems, settings)
            summaries = {s: cluster_summary(h, settings)
                         for s, h in event.items()}
            clusters = [c for cs in summaries.values() for c in cs]
            if args.entries is None:
                if not clusters:
                    continue
                if max(c[1] for c in clusters) < args.min_size:
                    continue
                if args.matched_only and not any(c[2] >= 0 for c in clusters):
                    continue
            nhit = sum(len(h["Z"]) for h in event.values())
            meta = (f"{tag} — {Path(path).name}, entry {i}\n"
                    f"{nhit} rechits, {len(clusters)} clusters")
            out_path = outdir / f"evd_{tag}_{Path(path).stem}_ev{i}.png"
            draw_event(event, meta, out_path, settings, zoom=args.zoom,
                       with_etaphi=args.with_etaphi, is_data=is_data)
            print(f"{out_path}  ({nhit} hits, "
                  + ", ".join(f"{SYSTEMS[s]['label']}:{len(cs)}"
                              for s, cs in summaries.items() if cs) + ")")
            ndrawn += 1
        if ndrawn >= args.nevents:
            break

    if ndrawn == 0:
        print("no event passed the selection — loosen --min-size / --matched-only "
              "or scan more files with --nfiles")


if __name__ == "__main__":
    main()
