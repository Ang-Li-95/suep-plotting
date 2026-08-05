"""Root-cause validation: the toy SuepShower with the pz typo vs the samples.

Left  -- cos(theta*_z) in the data, next to the toy run with and without the typo.
Right -- the same folded into |cos(theta*_z)|, showing that folding removes the
         forward/backward asymmetry but leaves the angular distribution wrong.
"""
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")
plt.rcParams["figure.dpi"] = 140
C1, C2, GREY = "tab:blue", "tab:red", "0.35"

ns = {}
exec(compile(open("toy.py").read().split('print("toy SuepShower')[0], "toy.py", "exec"), ns)

toy = {}
for buggy in (True, False):
    with contextlib.redirect_stdout(io.StringIO()):
        toy[buggy] = ns["run"]("x", 2.0, 1.0, buggy, nev=5000, seed=11)

data_cz = np.load("gen_asymmetry.npz")["T1_cos_z"].astype(float)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

e = np.linspace(-1, 1, 41)
c = 0.5 * (e[1:] + e[:-1])
hd, _ = np.histogram(data_cz, bins=e, density=True)
hb, _ = np.histogram(toy[True][1], bins=e, density=True)
hf, _ = np.histogram(toy[False][1], bins=e, density=True)
axes[0].errorbar(c, hd, fmt="o", ms=5, color="k", label="samples (T = 1)")
hep.histplot(hb, e, ax=axes[0], color=C2, lw=2,
             label=r"toy, $p_z = p\sin\theta$  (as shipped)")
hep.histplot(hf, e, ax=axes[0], color=GREY, ls="--", lw=2,
             label=r"toy, $p_z = p\cos\theta$  (fixed)")
axes[0].set_xlabel(r"$\cos\theta^*_z$")
axes[0].set_ylabel("normalised")
axes[0].set_ylim(0, max(hd.max(), hb.max()) * 1.45)
axes[0].legend(fontsize=12, loc="upper left")
hep.cms.label(data=False, com=13.6, ax=axes[0], fontsize=18)

e2 = np.linspace(0, 1, 21)
c2 = 0.5 * (e2[1:] + e2[:-1])
fd, _ = np.histogram(np.abs(data_cz), bins=e2, density=True)
fb, _ = np.histogram(np.abs(toy[True][1]), bins=e2, density=True)
axes[1].errorbar(c2, fd, fmt="o", ms=5, color="k", label="samples, folded")
hep.histplot(fb, e2, ax=axes[1], color=C2, lw=2, label="toy as shipped, folded")
axes[1].axhline(1.0, color=GREY, ls="--", lw=2, label="isotropic")
axes[1].set_xlabel(r"$|\cos\theta^*_z|$")
axes[1].set_ylabel("normalised")
axes[1].set_ylim(0, 2.0)
axes[1].legend(fontsize=12, loc="upper right")
hep.cms.label(data=False, com=13.6, ax=axes[1], fontsize=18)

fig.tight_layout()
fig.savefig("fig10_rootcause.png", bbox_inches="tight")

print("data  <cos>=%+.4f  toy shipped %+.4f  toy fixed %+.4f"
      % (data_cz.mean(), toy[True][1].mean(), toy[False][1].mean()))
print("folded shape, data/isotropic:", np.round(fd, 3))
print("max deviation from isotropic after folding: %.0f%%"
      % (100 * max(abs(fd - 1).max(), abs(fb - 1).max())))
