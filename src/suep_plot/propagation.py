"""Distance from a muon-system rechit to a particle from the interaction point.

Rechit ``Eta`` / ``Phi`` are detector positions, not directions from the
vertex, so a charged particle's own hits sit away from its production
direction in phi: a muon turns by ~1.71/pT rad by the coil radius in the 3.8 T
field.  :func:`bend_dphi` predicts that turn with the 3D helix model the DY
noise study fitted and validated, and :func:`min_dr_bent` uses it to measure
how far each rechit is from the nearest charged particle.

This is a port of ``Generation/DYNoiseStudy/analysis/punchthrough.py``
(``bend_dphi_z``, ``is_endcap_hit``, ``reachable_muons``; README "The
propagation now uses the hit z").  The study loops over events with
(hits x particles) matrices; here every function is element-wise, so one call
covers all hit-particle pairs of a chunk.  The arithmetic keeps the study's
order and dtypes (float32 kinematics, integer charge) on purpose: the overlay
cleaning is validated against the study's numbers, and a "tidied" expression
can move a hit across the dR boundary.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

B_FIELD = 3.8   # T
R_COIL = 3.0    # m: the solenoid radius, where the uniform field ends
# Effective half-length of the field [m], fitted by the study's
# bending_check.py on single-muon events: it absorbs the field falling off
# before the coil end (6.25 m) and the field in the endcap iron.
Z_END = 4.5

# Endcap stations are planes in z, barrel stations cylinders in r.  CSC and RPC
# endcap sit at |z| >= 6.8 m or (ME1/1) at r < 2.6 m; DT and RPC barrel at
# r >= 4.1 m and |z| <= 6.5 m.
ENDCAP_MIN_ABS_Z = 6.6   # m
ENDCAP_MAX_R = 4.0       # m

# Where the study measured that a status-1 gen muon stops leaving rechits
# (muon_reach.py): beyond |eta| 2.5, and below these momenta.
REACH_MAX_ABS_ETA = 2.5
BARREL_MAX_ABS_ETA = 0.9
OVERLAP_MAX_ABS_ETA = 1.2
BARREL_MIN_PT = 3.0      # GeV, |eta| < 0.9: the yoke stops softer muons
OVERLAP_MIN_PT = 2.5     # GeV, 0.9 <= |eta| < 1.2
ENDCAP_MIN_P = 2.5       # GeV, total momentum, |eta| >= 1.2
MAX_ABS_ETA_FOR_P = 6.0  # cap on |eta| when computing p = pt cosh(eta)

MUON_PDGID = 13
FINAL_STATE = 1


def delta_phi(a, b):
    """a - b, wrapped into [-pi, pi)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def is_endcap_hit(r_m, z_m):
    """Whether a hit at transverse radius *r_m* and *z_m* [m] is in an endcap station."""
    return (np.abs(z_m) > ENDCAP_MIN_ABS_Z) | (r_m < ENDCAP_MAX_R)


def bend_dphi(r_m, z_m, eta, pt, charge, z_end=Z_END):
    """Azimuth of the hit position minus the particle's production azimuth.

    Element-wise over equal-length arrays: one entry per (hit, particle) pair,
    with the hit's *r_m* / *z_m* [m] and the particle's *eta*, *pt* [GeV] and
    integer *charge*.

    The particle follows a helix of transverse radius rho = pt / (0.3 B): its
    position azimuth turns by psi/2 and its direction by psi, in the sense -q,
    where psi grows along the 3D path (psi = |z| / (rho |sinh eta|)).  It stays
    in the field until it leaves the cylinder r < R_COIL, |z| < *z_end* through
    the wall or the end face, whichever comes first, then goes straight.  An
    endcap hit is located on the trajectory by its z, a barrel hit by its r.
    Neutral particles do not bend.
    """
    r = np.asarray(r_m)
    za = np.abs(np.asarray(z_m))
    endcap = is_endcap_hit(r, za)
    neutral = charge == 0
    s = -charge
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rho = np.where(neutral, np.inf, pt / (0.3 * B_FIELD))
        ash = np.abs(np.sinh(np.clip(eta, -10, 10)))
        # turning angle at which the particle leaves the field
        psi_r = np.where(R_COIL < 2 * rho,
                         2 * np.arcsin(np.clip(R_COIL / (2 * rho), 0, 1)), np.inf)
        psi_z = np.where(ash > 0, z_end / (rho * ash), np.inf)
        psi_e = np.minimum(psi_r, psi_z)
        # turning angle at the hit, if the hit is inside the field
        psi_hz = za / (rho * ash)
        psi_hr = 2 * np.arcsin(np.clip(r / (2 * rho), 0, 1))
        psi_h = np.where(endcap, psi_hz, psi_hr)
        inside = psi_h <= psi_e
        # A soft particle can turn more than once before leaving the field:
        # z grows with the unwrapped psi, the position on the circle repeats
        # every 2 pi.
        two_pi = 2 * np.pi
        psi_hw = np.mod(psi_h, two_pi)
        psi_ew = np.mod(psi_e, two_pi)
        # outside: exit point and direction, then a straight line to the hit
        r0 = 2 * rho * np.sin(psi_ew / 2)
        u = s * psi_ew / 2
        d = s * psi_ew
        t_end = (za - rho * ash * psi_e) / ash
        disc = r ** 2 - (r0 * np.sin(psi_ew / 2)) ** 2
        t_bar = -r0 * np.cos(psi_ew / 2) + np.sqrt(np.clip(disc, 0, None))
        t = np.clip(np.where(endcap, t_end, t_bar), 0, None)
        px = r0 * np.cos(u) + t * np.cos(d)
        py = r0 * np.sin(u) + t * np.sin(d)
        out = np.where(inside, s * psi_hw / 2, np.arctan2(py, px))
    return np.where(neutral | ~np.isfinite(out), 0.0, out)


def gen_muon_charge(pdg_id):
    """Charge of a gen muon from its pdgId: +13 is the mu-."""
    return -np.sign(pdg_id)


def reachable_gen_muons(gen):
    """The status-1 gen muons that can reach the muon chambers.

    *gen* is a GenPart-like collection (``pdgId``, ``status``, ``pt``, ``eta``,
    ``phi``).  Only muons that cannot reach are removed; the study found this
    keeps every muon-made rechit.  Returns the selected particles with a
    ``charge`` field.
    """
    eta, pt = gen.eta, gen.pt
    abs_eta = np.abs(eta)
    # |eta| capped as in the study, so the beam-line particles' cosh does not
    # overflow; it cannot change the decision, which requires |eta| < 2.5.
    momentum = pt * np.cosh(np.minimum(abs_eta, MAX_ABS_ETA_FOR_P))
    can_reach = ak.where(abs_eta < BARREL_MAX_ABS_ETA, pt > BARREL_MIN_PT,
                         ak.where(abs_eta < OVERLAP_MAX_ABS_ETA, pt > OVERLAP_MIN_PT,
                                  momentum > ENDCAP_MIN_P))
    keep = ((np.abs(gen.pdgId) == MUON_PDGID) & (gen.status == FINAL_STATE)
            & (abs_eta < REACH_MAX_ABS_ETA) & can_reach)
    muons = gen[keep]
    return ak.zip({"eta": muons.eta, "phi": muons.phi, "pt": muons.pt,
                   "charge": gen_muon_charge(muons.pdgId)})


def min_dr(hits, objects):
    """Per rechit, the plain dR to the nearest object in its event; inf if none.

    *hits* carry ``Eta`` / ``Phi``, *objects* ``eta`` / ``phi`` (jets: their
    axis is not bent).
    """
    return _min_over_pairs(hits, ("Eta", "Phi"), objects, ("eta", "phi"), _plain_dr)


def min_dr_bent(hits, tracks):
    """Per rechit, the dR to the nearest charged particle, its phi bent to the hit.

    *hits* carry ``X`` / ``Y`` / ``Z`` [cm] and ``Eta`` / ``Phi``; *tracks*
    carry ``eta``, ``phi``, ``pt`` and an integer ``charge``.  inf in events
    with no track.
    """
    return _min_over_pairs(hits, ("X", "Y", "Z", "Eta", "Phi"),
                           tracks, ("eta", "phi", "pt", "charge"), _bent_dr)


def _plain_dr(hit, obj):
    deta = hit["Eta"] - obj["eta"]
    return np.sqrt(deta ** 2 + delta_phi(hit["Phi"], obj["phi"]) ** 2)


def _bent_dr(hit, trk):
    r_m = np.sqrt(hit["X"] ** 2 + hit["Y"] ** 2) / 100.0   # cm -> m
    z_m = hit["Z"] / 100.0
    phi_at_hit = trk["phi"] + bend_dphi(r_m, z_m, trk["eta"], trk["pt"], trk["charge"])
    deta = hit["Eta"] - trk["eta"]
    return np.sqrt(deta ** 2 + delta_phi(hit["Phi"], phi_at_hit) ** 2)


def _min_over_pairs(hits, hit_fields, objects, object_fields, distance):
    """Apply *distance* to every (hit, object) pair of each event; min per hit.

    Only the named fields are paired, and the pairs are flattened to plain
    numpy arrays, so *distance* runs once per chunk on arrays of the input
    dtypes.
    """
    pairs = ak.cartesian({"hit": ak.zip({f: hits[f] for f in hit_fields}),
                          "obj": ak.zip({f: objects[f] for f in object_fields})},
                         axis=1, nested=True)
    hits_per_event = ak.num(pairs, axis=1)
    objects_per_hit = ak.flatten(ak.num(pairs, axis=2))

    def flat(side, field):
        return ak.to_numpy(ak.flatten(pairs[side][field], axis=None))

    hit = {f: flat("hit", f) for f in hit_fields}
    obj = {f: flat("obj", f) for f in object_fields}
    pair_dr = distance(hit, obj)

    per_hit = ak.unflatten(ak.unflatten(pair_dr, objects_per_hit), hits_per_event)
    return ak.fill_none(ak.min(per_hit, axis=2), np.inf)
