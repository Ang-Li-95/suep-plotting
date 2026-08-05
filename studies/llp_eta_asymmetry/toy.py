"""Toy re-implementation of CMSSW SuepShower, with and without the pz typo.

Reproduces GeneratorInterface/Pythia8Interface/src/SuepShower.cc:
  generateFourVector()  -> thermal |p|, isotropic angles, compose the 4-vector
  generateShower()      -> fill until E > mMediator, subtract the mean
                           3-momentum, then rescale |p| to conserve energy

The only difference between the two runs is the z component of line 158:
  buggy: pz = p * sin(theta)      fixed: pz = p * cos(theta)
"""
import numpy as np
from scipy.optimize import brentq

M_MED = 125.0


def sample_p(n, m_over_T, rng):
    """|p|/m from f(x) = x^2 exp(-(m/T) x^2 / (1 + sqrt(1+x^2))), by rejection."""
    f = lambda x: x ** 2 * np.exp(-m_over_T * x ** 2 / (1 + np.sqrt(1 + x ** 2)))
    grid = np.linspace(0, 40, 20000)
    fmax = f(grid).max()
    out = np.empty(0)
    while len(out) < n:
        x = rng.uniform(0, 40, 4 * n)
        out = np.concatenate([out, x[rng.uniform(0, fmax, len(x)) < f(x)]])
    return out[:n]


def one_event(m_dark, m_over_T, rng, buggy):
    # generateFourVector(), in blocks until the energy budget is filled
    # one daughter at a time, exactly as the while loop in generateShower()
    pool_mom = sample_p(200, m_over_T, rng) * m_dark
    pool_phi = rng.uniform(0, 2 * np.pi, 200)
    pool_th = np.arccos(2 * rng.uniform(0, 1, 200) - 1)
    esum, k, px, py, pz = 0.0, 0, [], [], []
    while esum < M_MED or k < 2:
        mom, phi, th = pool_mom[k], pool_phi[k], pool_th[k]
        px.append(mom * np.cos(phi) * np.sin(th))
        py.append(mom * np.sin(phi) * np.sin(th))
        pz.append(mom * (np.sin(th) if buggy else np.cos(th)))
        esum += np.sqrt(mom ** 2 + m_dark ** 2)
        k += 1
    px, py, pz = np.array(px), np.array(py), np.array(pz)

    # "Reballance momenta": subtract the mean 3-momentum from every daughter
    px, py, pz = px - px.mean(), py - py.mean(), pz - pz.mean()

    # "balance energy": common rescale of |p| so that sum(E) = m_mediator
    p2 = px ** 2 + py ** 2 + pz ** 2
    g = lambda s: np.sqrt(s * s * p2 + m_dark ** 2).sum() - M_MED
    lo, hi = 0.0, 2.0
    while g(lo) * g(hi) > 0:
        lo, hi = hi, hi * 2
    s = brentq(g, lo, hi)
    px, py, pz = s * px, s * py, s * pz
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return p, pz / p, px / p, py / p


def run(label, m_dark, temperature, buggy, nev=2000, seed=1):
    rng = np.random.default_rng(seed)
    P, CZ, CX, CY, N = [], [], [], [], []
    for _ in range(nev):
        p, cz, cx, cy = one_event(m_dark, m_dark / temperature, rng, buggy)
        P.append(p); CZ.append(cz); CX.append(cx); CY.append(cy); N.append(len(p))
    P, CZ, CX, CY = (np.concatenate(a) for a in (P, CZ, CX, CY))
    print(f"{label:28s} <n>={np.mean(N):5.1f}  <p*>={P.mean():5.2f}  "
          f"<cos_z>={CZ.mean():+.4f}  frac(cos_z<0)={(CZ < 0).mean():.4f}  "
          f"corr={np.corrcoef(P, CZ)[0, 1]:+.3f}  "
          f"<cos_x>={CX.mean():+.4f} <cos_y>={CY.mean():+.4f}")
    return P, CZ


print("toy SuepShower, mMediator = 125 GeV\n")
for T in (1.0, 2.0):
    for buggy in (True, False):
        tag = "pz = p*sin(theta)  [as shipped]" if buggy else "pz = p*cos(theta)  [fixed]"
        run(f"T={T:g}  {tag}", 2.0, T, buggy)

print("\nmeasured in the samples:")
print("T=1: <n>= 31.0  <p*>= 3.41  <cos_z>=-0.1822  frac= 0.5749  corr=+0.723")
print("T=2: <n>= 19.2                <cos_z>=-0.1861  frac= 0.5789")

P, CZ = run("T=1 buggy, for the shape", 2.0, 1.0, True, nev=6000, seed=7)
h, _ = np.histogram(CZ, bins=10, range=(-1, 1))
print("\ntoy   cos_z shape:", (h / h.sum()).round(4))
print("data  cos_z shape: [0.1574 0.1041 0.0957 0.1008 0.1168 0.1394 "
      "0.1638 0.1119 0.0092 0.0008]")
np.savez_compressed("toy_buggy.npz", p=P, cz=CZ)
