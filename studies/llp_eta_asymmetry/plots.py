"""Diagnostic histograms for the LLP eta asymmetry in the SUEPs_Gen2 samples.

Each figure is one step of the argument, in the order it is made on the slides:
observation -> parent is clean -> momentum closes -> the anisotropy is in the
SUEP rest frame, along a fixed lab axis, correlated with p* -> visible with no
boost at all -> it propagates to the reconstructed clusters.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")
plt.rcParams["figure.dpi"] = 140

D = np.load("gen_asymmetry.npz")
REPO = "/users/ang.li/public/SUEP/suep-plotting/output_mds_llp/"
C1, C2 = "tab:blue", "tab:red"


def label(ax):
    hep.cms.label(data=False, com=13.6, ax=ax, fontsize=18)


def mirror_panel(ax_hi, ax_lo, x, bins, rng, color, name):
    """h(x) overlaid with h(-x), plus the ratio -- an eye-free asymmetry test."""
    h, e = np.histogram(x, bins=bins, range=rng)
    hm, _ = np.histogram(-x, bins=bins, range=rng)
    c = 0.5 * (e[1:] + e[:-1])
    hep.histplot(h, e, ax=ax_hi, color=color, label=f"{name}")
    hep.histplot(hm, e, ax=ax_hi, color="k", ls="--", label=f"{name}, mirrored")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(hm > 0, h / hm, np.nan)
        er = r * np.sqrt(1 / np.maximum(h, 1) + 1 / np.maximum(hm, 1))
    # drop bins where either side is too empty for the ratio to mean anything
    ok = (h >= 20) & (hm >= 20)
    ax_lo.errorbar(c[ok], r[ok], yerr=er[ok], fmt="o", ms=3, color=color)
    ax_lo.axhline(1.0, color="k", lw=1)
    ax_hi.set_ylim(0, max(ax_hi.get_ylim()[1], h.max() * 1.45))
    ax_lo.set_ylim(0.7, 1.3)
    ax_lo.set_ylabel(r"$h(x)/h(-x)$", fontsize=15)


def two_row(figsize=(8, 7)):
    fig, (hi, lo) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    return fig, hi, lo


# ── 1. the observation: gen LLP eta is not symmetric ──────────────────
fig, hi, lo = two_row()
for tag, col, nm in [("T1", C1, "T=1"), ("T2", C2, "T=2")]:
    mirror_panel(hi, lo, D[f"{tag}_lab_eta"], 60, (-3, 3), col, nm)
hi.set_ylabel("dark mesons")
hi.legend(fontsize=13, ncol=2, loc="upper center")
label(hi)
lo.set_xlabel(r"gen dark meson $\eta$ (lab)")
for tag, nm in [("T1", "T=1"), ("T2", "T=2")]:
    e = D[f"{tag}_lab_eta"]
    print(f"[1] {nm}: <eta>={e.mean():+.4f}  frac(eta<0)={(e < 0).mean():.4f}")
fig.savefig("fig1_lab_eta.png", bbox_inches="tight")

# ── 2. the parent Higgs is clean ──────────────────────────────────────
fig, hi, lo = two_row()
mirror_panel(hi, lo, D["T1_h_eta"], 50, (-4, 4), C1, "H (T=1)")
hi.set_ylabel("events")
hi.legend(fontsize=13, loc="upper right")
label(hi)
lo.set_xlabel(r"gen Higgs $\eta$ (status 62)")
h = D["T1_h_eta"]
hi.text(0.04, 0.95, va="top", s=r"$\langle\eta_H\rangle$ = %+.3f $\pm$ %.3f" % (h.mean(), h.std() / np.sqrt(len(h)))
        + "\n" + r"frac($\eta_H<0$) = %.4f" % (h < 0).mean(),
        transform=hi.transAxes, fontsize=15)
print(f"[2] <eta_H>={h.mean():+.4f} +- {h.std()/np.sqrt(len(h)):.4f}  frac<0={(h<0).mean():.4f}")
fig.savefig("fig2_higgs_eta.png", bbox_inches="tight")

# ── 3. 4-momentum closure: it is not a broken boost ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, col, nm in [("d_px", "tab:green", r"$\Delta p_x$"),
                   ("d_py", "tab:orange", r"$\Delta p_y$"),
                   ("d_pz", C1, r"$\Delta p_z$"), ("d_E", C2, r"$\Delta E$")]:
    v = D[f"T1_{k}"]
    hep.histplot(*np.histogram(v, bins=80, range=(-8, 8)), ax=axes[0],
                 color=col, label=f"{nm}: {v.mean():+.3f} GeV")
axes[0].set_xlabel(r"$\sum p(\mathrm{mesons}) - p(H)$  [GeV]")
axes[0].set_ylabel("events")
axes[0].legend(fontsize=12)
axes[0].set_yscale("log")
label(axes[0])

m = D["T1_s_mass"]
hep.histplot(*np.histogram(m, bins=60, range=(124.9, 125.1)), ax=axes[1], color=C1)
axes[1].set_xlabel(r"$m$(dark-meson system)  [GeV]")
axes[1].set_ylabel("events")
axes[1].text(0.04, 0.85, r"mean = %.3f GeV" % m.mean(), transform=axes[1].transAxes,
             fontsize=15)
label(axes[1])
print(f"[3] <dpz>={D['T1_d_pz'].mean():+.4f}  <m_SUEP>={m.mean():.4f}")
fig.tight_layout()
fig.savefig("fig3_closure.png", bbox_inches="tight")

# ── 4. rest frame: isotropic in x,y -- not in z ────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for k, col, nm in [("cos_x", "tab:green", r"$\cos\theta^*_x$"),
                   ("cos_y", "tab:orange", r"$\cos\theta^*_y$"),
                   ("cos_z", C1, r"$\cos\theta^*_z$")]:
    v = D[f"T1_{k}"]
    hep.histplot(*np.histogram(v, bins=40, range=(-1, 1)), ax=ax, color=col,
                 label=r"%s   $\langle\cdot\rangle$ = %+.4f" % (nm, v.mean()))
ax.set_xlabel(r"direction cosine in the dark-meson rest frame")
ax.set_ylabel("dark mesons")
ax.legend(fontsize=13, loc="upper left")
ax.set_ylim(0, None)
label(ax)
cz = D["T1_cos_z"]
print(f"[4] <cos_x>={D['T1_cos_x'].mean():+.5f} <cos_y>={D['T1_cos_y'].mean():+.5f} "
      f"<cos_z>={cz.mean():+.5f}  sig={abs(cz.mean())/(cz.std()/np.sqrt(len(cz))):.0f}")
fig.savefig("fig4_restframe_cos.png", bbox_inches="tight")

# ── 5. the axis is the fixed lab z, not the parent direction ──────────
fig, ax = plt.subplots(figsize=(8, 6))
pe = D["T1_parent_eta"]
for m_, col, nm in [(pe < 0, C1, r"$\eta_H < 0$"), (pe > 0, C2, r"$\eta_H > 0$")]:
    v = D["T1_cos_z"][m_]
    hep.histplot(*np.histogram(v, bins=40, range=(-1, 1), density=True), ax=ax,
                 color=col, label=r"%s:  $\langle\cos\theta^*_z\rangle$ = %+.4f" % (nm, v.mean()))
ax.set_xlabel(r"$\cos\theta^*_z$  (dark-meson rest frame)")
ax.set_ylabel("normalised")
ax.legend(fontsize=13, loc="upper left")
ax.set_ylim(0, None)
label(ax)
print("[5] <cos_z|etaH<0>=%+.4f  <cos_z|etaH>0>=%+.4f"
      % (D["T1_cos_z"][pe < 0].mean(), D["T1_cos_z"][pe > 0].mean()))
fig.savefig("fig5_cos_by_parent.png", bbox_inches="tight")

# ── 6. soft mesons go -z, hard mesons go +z ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ps, cz = D["T1_p_star"], D["T1_cos_z"]
h2, xe, ye = np.histogram2d(ps, cz, bins=(60, 50), range=((0, 15), (-1, 1)))
axes[0].pcolormesh(xe, ye, h2.T, cmap="viridis")
axes[0].set_xlabel(r"$p^*$ [GeV]")
axes[0].set_ylabel(r"$\cos\theta^*_z$")
label(axes[0])

edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20])
idx = np.digitize(ps, edges) - 1
ok = (idx >= 0) & (idx < len(edges) - 1)
mean = np.array([cz[ok & (idx == i)].mean() for i in range(len(edges) - 1)])
err = np.array([cz[ok & (idx == i)].std() / np.sqrt(max((ok & (idx == i)).sum(), 1))
                for i in range(len(edges) - 1)])
axes[1].errorbar(0.5 * (edges[1:] + edges[:-1]), mean, yerr=err,
                 xerr=0.5 * np.diff(edges), fmt="o", color=C1)
axes[1].axhline(0, color="k", lw=1, ls="--")
axes[1].set_xlabel(r"$p^*$ [GeV]")
axes[1].set_ylabel(r"$\langle\cos\theta^*_z\rangle$")
label(axes[1])
w = (cz * ps).sum() / ps.sum()
print("[6] profile:", np.round(mean, 3), " momentum-weighted <cos_z> = %.6f" % w)
fig.tight_layout()
fig.savefig("fig6_cos_vs_pstar.png", bbox_inches="tight")

# ── 7. no boost needed: SUEP system already at rest ───────────────────
fig, hi, lo = two_row()
sel = np.abs(D["T1_parent_spz"]) < 10
x = D["T1_lab_eta"][sel]
mirror_panel(hi, lo, x, 40, (-3, 3), C1, r"$|p_z^{\rm SUEP}| < 10$ GeV")
hi.set_ylabel("dark mesons")
hi.legend(fontsize=12, loc="upper right")
hi.text(0.04, 0.95, va="top", s=r"$\langle\eta\rangle$ = %+.3f,  frac($\eta<0$) = %.3f"
        % (x.mean(), (x < 0).mean()) + "\n%d dark mesons, no boost applied"
        % len(x), transform=hi.transAxes, fontsize=14)
label(hi)
lo.set_xlabel(r"gen dark meson $\eta$ (lab)")
print(f"[7] at-rest events: <eta>={x.mean():+.4f} frac<0={(x < 0).mean():.4f} n={len(x)}")
fig.savefig("fig7_no_boost.png", bbox_inches="tight")

# ── 8. it propagates to the reconstructed clusters ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, xlabel in [(axes[0], "csc_cluster_eta", r"CSC cluster $\eta$"),
                        (axes[1], "dt_cluster_eta", r"DT cluster $\eta$")]:
    for sample, col, nm in [("suep_temp1", C1, "T=1"), ("suep_temp2", C2, "T=2")]:
        with open(f"{REPO}{sample}.pkl", "rb") as fh:
            h = pickle.load(fh)["histograms"][key]
        # values() carries a leading dataset category axis; collapse it
        e = h.axes[-1].edges
        v = np.asarray(h.values()).reshape(-1, len(e) - 1).sum(axis=0)
        v, e = v.reshape(-1, 2).sum(axis=1), e[::2]
        c = 0.5 * (e[1:] + e[:-1])
        a = (v[c < 0].sum() - v[c > 0].sum()) / v.sum()
        hep.histplot(v / v.sum(), e, ax=ax, color=col,
                     label=r"%s:  $A$ = %+.3f" % (nm, a))
        print(f"[8] {key} {nm}: A={a:+.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalised")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)   # headroom so the legend clears the data
    ax.legend(fontsize=13, loc="upper left")
    label(ax)
fig.tight_layout()
fig.savefig("fig8_reco_clusters.png", bbox_inches="tight")

# ── 9. it is not a frame error: nothing restores isotropy ─────────────
# Scan a longitudinal boost and a longitudinal momentum offset applied to the
# rest-frame momenta.  If the anisotropy were bookkeeping (wrong frame, or a
# momentum-balancing shift), one of these would flatten it out.  Neither does.
cz, ps = D["T1_cos_z"].astype(float), D["T1_p_star"].astype(float)
pz, pt2 = ps * cz, np.maximum(ps ** 2 - (ps * cz) ** 2, 0)
E = np.sqrt(ps ** 2 + 2.0 ** 2)


def scan(transform, values):
    out = []
    for v in values:
        pz2, p2 = transform(v)
        c2 = pz2 / p2
        out.append((c2.mean(), np.corrcoef(p2, c2)[0, 1]))
    return np.array(out)


betas = np.linspace(-0.6, 0.4, 21)
def boost(b):
    g = 1 / np.sqrt(1 - b ** 2)
    pz2 = g * (pz - b * E)
    return pz2, np.sqrt(pz2 ** 2 + pt2)


offs = np.linspace(-1.5, 3.0, 19)
def shift(d):
    return pz - d, np.sqrt((pz - d) ** 2 + pt2)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, res, xlabel in [
        (axes[0], betas, scan(boost, betas), r"applied boost $\beta_z$"),
        (axes[1], offs, scan(shift, offs), r"applied offset $\Delta p_z$ [GeV]")]:
    ax.plot(vals, res[:, 0], "o-", color=C1, label=r"$\langle\cos\theta^*_z\rangle$")
    ax.plot(vals, res[:, 1], "s-", color=C2, label=r"corr($p^*$, $\cos\theta^*_z$)")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.8, 1.0)
    ax.legend(fontsize=13, loc="lower left")
    label(ax)
axes[0].set_ylabel("value")
print("[9] corr(p*,cos*) range over both scans: %.2f - %.2f"
      % (min(scan(boost, betas)[:, 1].min(), scan(shift, offs)[:, 1].min()),
         max(scan(boost, betas)[:, 1].max(), scan(shift, offs)[:, 1].max())))
fig.tight_layout()
fig.savefig("fig9_not_a_frame_error.png", bbox_inches="tight")

print("done")
