"""suep_plot.propagation: the DY noise study's helix model, ported element-wise."""

import math

import awkward as ak
import numpy as np
import pytest

from suep_plot.propagation import (bend_dphi, min_dr, min_dr_bent,
                                   reachable_gen_muons)

# Hits (r, z) [m]: barrel MB1 and MB4, endcap ME1/1, ME2, ME4.
HIT_R = np.array([4.2, 7.0, 1.5, 3.5, 6.5], dtype=np.float32)
HIT_Z = np.array([1.0, -5.0, 6.8, -8.3, 10.2], dtype=np.float32)
# Particles: eta, pt [GeV], charge; the last one is neutral.
P_ETA = np.array([0.3, -0.8, 2.0, -1.6, 2.3, 0.0], dtype=np.float32)
P_PT = np.array([5.0, 20.0, 3.0, 100.0, 1.2, 8.0], dtype=np.float32)
P_Q = np.array([1, -1, -1, 1, 1, 0], dtype=np.int32)

# punchthrough.bend_dphi_z(HIT_R, HIT_Z, P_ETA, P_PT, P_Q) from
# Generation/DYNoiseStudy/analysis (2026-10-15 version), printed at full
# precision on 2026-11-19.  Rows are hits, columns particles.
STUDY_BEND = np.array([
    [-0.45130577667839245, 0.11009957548879529, 0.40306793361481535,
     -0.016725154719340427, -0.7775671086357165, 0.0],
    [-0.5509874470568668, 0.13455797250277726, 0.4304541303262145,
     -0.018673035406421398, -0.8129512989893438, 0.0],
    [-0.6516705993943619, 0.13765751980772664, 0.3158101240270175,
     -0.014449518360433903, -0.5815010970506582, 0.0],
    [-0.6601257665926406, 0.1437271741874279, 0.3441704590420307,
     -0.01574086269056162, -0.6343279638082646, 0.0],
    [-0.6672438895355712, 0.1488509794079101, 0.3680838459902661,
     -0.016831327726109173, -0.6787058722565025, 0.0],
])


def test_bend_reproduces_the_study_bit_for_bit():
    hit, particle = np.meshgrid(np.arange(len(HIT_R)), np.arange(len(P_ETA)),
                                indexing="ij")
    hit, particle = hit.ravel(), particle.ravel()

    bend = bend_dphi(HIT_R[hit], HIT_Z[hit], P_ETA[particle], P_PT[particle],
                     P_Q[particle])

    assert np.array_equal(bend, STUDY_BEND.ravel())


def test_opposite_charges_bend_opposite_ways():
    args = (HIT_R[:1], HIT_Z[:1], P_ETA[:1], P_PT[:1])
    plus = bend_dphi(*args, np.array([1], dtype=np.int32))
    minus = bend_dphi(*args, np.array([-1], dtype=np.int32))
    assert plus[0] == pytest.approx(-minus[0])
    assert plus[0] < 0   # a positive charge turns clockwise in the +z field


def _barrel_hit(phi, r_cm=420.0, z_cm=100.0):
    return {"X": r_cm * math.cos(phi), "Y": r_cm * math.sin(phi), "Z": z_cm,
            "Eta": math.asinh(z_cm / r_cm), "Phi": phi}


def _events(hits, tracks):
    """One event: lists of hit and track dicts -> jagged records (float32)."""
    def collection(rows, fields, dtype):
        return ak.zip({f: ak.values_astype(ak.Array([[row[f] for row in rows]]),
                                           dtype.get(f, np.float32))
                       for f in fields})
    hit_arr = collection(hits, ("X", "Y", "Z", "Eta", "Phi"), {})
    track_arr = collection(tracks, ("eta", "phi", "pt", "charge"), {"charge": np.int32})
    return hit_arr, track_arr


def test_a_soft_muon_reaches_its_hit_only_once_bent():
    eta = math.asinh(100.0 / 420.0)        # the muon points at the hit's z/r
    muon = {"eta": eta, "phi": 0.0, "pt": 5.0, "charge": 1}
    bend = float(bend_dphi(np.float32([4.2]), np.float32([1.0]), np.float32([eta]),
                           np.float32([5.0]), np.int32([1]))[0])
    at_bent_position = _barrel_hit(bend)
    at_production_phi = _barrel_hit(0.0)
    hits, tracks = _events([at_bent_position, at_production_phi], [muon])

    bent = ak.to_list(min_dr_bent(hits, tracks)[0])
    plain = ak.to_list(min_dr(hits, tracks)[0])

    assert bent[0] < 0.01 and bent[1] > 0.4    # bent: only the bent position is near
    assert plain[0] > 0.4 and plain[1] < 0.01  # plain dR gets it backwards


def test_no_object_means_infinitely_far():
    hits, tracks = _events([_barrel_hit(0.0)], [])
    assert ak.to_list(min_dr_bent(hits, tracks)) == [[math.inf]]
    assert ak.to_list(min_dr(hits, tracks)) == [[math.inf]]


def _gen(*particles):
    """One event of GenPart rows (pdgId, status, pt, eta)."""
    fields = ("pdgId", "status", "pt", "eta")
    data = {f: [[p[i] for p in particles]] for i, f in enumerate(fields)}
    data["phi"] = [[0.0] * len(particles)]
    return ak.Array(data)


@pytest.mark.parametrize("particle, reaches", [
    ((13, 1, 3.1, 0.5), True),     # barrel: pt > 3
    ((13, 1, 2.9, 0.5), False),
    ((-13, 1, 2.6, 1.0), True),    # overlap: pt > 2.5
    ((-13, 1, 2.4, 1.0), False),
    ((13, 1, 1.0, 2.0), True),     # endcap: p = pt cosh(eta) = 3.76 > 2.5
    ((13, 1, 0.5, 2.0), False),    # p = 1.88
    ((13, 1, 50.0, 2.6), False),   # beyond |eta| 2.5
    ((13, 2, 50.0, 0.5), False),   # not final state
    ((211, 1, 50.0, 0.5), False),  # not a muon
])
def test_reachable_gen_muon_edges(particle, reaches):
    muons = reachable_gen_muons(_gen(particle))
    assert ak.num(muons)[0] == int(reaches)


def test_gen_muon_charge_follows_the_pdgid_convention():
    muons = reachable_gen_muons(_gen((13, 1, 10.0, 0.0), (-13, 1, 10.0, 0.0)))
    assert ak.to_list(muons.charge[0]) == [-1, 1]   # +13 is the mu-
