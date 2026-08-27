"""Schematic defining the starred (rest-frame) quantities used on the slides."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

TEAL, ALERT, GREY, INK = "#1C7293", "#E4572E", "#8896A6", "#1B2A41"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax in (a, b):
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

arrow = lambda ax, p, q, c, lw=2.0, ls="-": ax.add_patch(FancyArrowPatch(
    p, q, arrowstyle="-|>", mutation_scale=15, lw=lw, color=c, ls=ls,
    shrinkA=0, shrinkB=0, zorder=3))

# ── lab frame ─────────────────────────────────────────────────────────
a.set_title("lab frame", fontsize=15, color=INK, pad=14)
a.plot([-1.1, 1.3], [0, 0], color=GREY, lw=1.2, zorder=0)
a.text(1.32, 0.0, "z", fontsize=14, color=GREY, va="center", ha="left")
arrow(a, (-1.05, 0), (-0.72, 0), GREY, 1.6)
arrow(a, (1.05, 0), (0.72, 0), GREY, 1.6)
a.add_patch(Circle((0.05, 0), 0.17, fc=TEAL, alpha=0.20, ec=TEAL, lw=1.4, zorder=2))
rng = np.random.default_rng(3)
for ang, ln in zip(rng.uniform(0, 2 * np.pi, 9), rng.uniform(0.3, 0.55, 9)):
    arrow(a, (0.05, 0), (0.05 + ln * np.cos(ang), ln * np.sin(ang)), TEAL, 1.3)
arrow(a, (0.05, 0), (0.78, 0.20), INK, 2.6)
a.text(0.80, 0.24, r"$\vec{p}_S$", fontsize=13.5, color=INK)
a.text(0.05, -0.82, "the dark mesons of one event;\ntheir 4-momenta sum to $S$",
       fontsize=11.5, color=INK, ha="center")

# ── rest frame ────────────────────────────────────────────────────────
b.set_title("rest frame of $S$  —  starred quantities", fontsize=15, color=INK, pad=14)
b.plot([-1.1, 1.3], [0, 0], color=GREY, lw=1.2, zorder=0)
b.text(1.32, 0, "z", fontsize=14, color=GREY, va="center", ha="left")
arrow(b, (0, -0.75), (0, 0.85), GREY, 1.2)
b.text(0.06, 0.88, "x, y", fontsize=12, color=GREY, va="bottom")
for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
    arrow(b, (0, 0), (0.34 * np.cos(ang), 0.34 * np.sin(ang)), TEAL, 1.1)

th = np.deg2rad(34)
L = 0.95
tip = (L * np.cos(th), L * np.sin(th))
arrow(b, (0, 0), tip, ALERT, 2.6)
b.plot([tip[0], tip[0]], [0, tip[1]], color=ALERT, lw=1.1, ls=":", zorder=3)
b.add_patch(Arc((0, 0), 0.86, 0.86, theta1=0, theta2=34, color=INK, lw=1.3))
b.text(0.52, 0.09, r"$\theta^*_z$", fontsize=15, color=INK)
b.text(tip[0] + 0.03, tip[1] + 0.05, r"$\vec{p}^{\,*}$ of one meson",
       fontsize=12.5, color=ALERT)
b.text(tip[0] + 0.02, tip[1] / 2, r"$p^*_z$", fontsize=12.5, color=ALERT, va="center")
b.text(0.1, -0.88, r"$p^* = |\vec{p}^{\,*}|$        "
                   r"$\cos\theta^*_z = p^*_z\,/\,p^*$",
       fontsize=13.5, color=INK, ha="center")

fig.tight_layout()
# the boost that defines the starred frame
fig.text(0.5, 0.60, r"$\Longrightarrow$", fontsize=22, color=GREY, ha="center")
fig.text(0.5, 0.525, "boost by $S$", fontsize=11, color=GREY, ha="center")
fig.savefig("fig0_notation.png", bbox_inches="tight")
print("wrote fig0_notation.png")
