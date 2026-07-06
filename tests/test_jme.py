"""JEC/JER tests on synthetic jets (skipped when jsonpog files are absent)."""

import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

JSONPOG = Path("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG")

pytestmark = pytest.mark.skipif(
    not (JSONPOG / "JME/2024_Summer24/jet_jerc.json.gz").exists()
    and not os.environ.get("CORRECTIONLIB_DATA"),
    reason="jsonpog-integration not available",
)


def make_events(gen_pt=None):
    """2 events x 2 jets; genJetIdx matches jet i to genjet i (or -1)."""
    def jag(vals):
        return ak.Array([vals[:2], vals[2:]])

    jets = ak.zip({
        "pt": jag([50.0, 30.0, 80.0, 20.0]),
        "eta": jag([0.5, -1.2, 1.8, 0.1]),
        "phi": jag([0.1, 2.0, -1.0, 3.0]),
        "mass": jag([8.0, 5.0, 12.0, 4.0]),
        "area": jag([0.5, 0.5, 0.5, 0.5]),
        "rawFactor": jag([0.1, 0.05, 0.12, 0.0]),
        "genJetIdx": ak.Array([[0, 1], [0, -1]]),
        "chEmEF": jag([0.0, 0.0, 0.0, 0.95]),   # last jet fails the EMF cut
        "neEmEF": jag([0.0, 0.0, 0.0, 0.0]),
    })
    if gen_pt is None:
        gen_pt = jag([50.0, 30.0, 80.0, 20.0])
    genjets = ak.zip({
        "pt": gen_pt,
        "eta": jag([0.5, -1.2, 1.8, 0.1]),
        "phi": jag([0.1, 2.0, -1.0, 3.0]),
    })
    rho = ak.zip({"fixedGridRhoFastjetAll": ak.Array([20.0, 25.0])})
    met = ak.zip({"pt": ak.Array([40.0, 60.0]), "phi": ak.Array([0.3, -2.0])})
    return ak.zip({"Jet": jets, "GenJet": genjets, "Rho": rho, "PuppiMET": met,
                   "event": ak.Array([11, 12])}, depth_limit=1)


def test_jec_variations_bracket_nominal():
    from suep_plot.jme import correct_jets

    ev = make_events()
    nom = ak.flatten(correct_jets(ev, smear=False).Jet.pt)
    up = ak.flatten(correct_jets(ev, smear=False, variation="jec_up").Jet.pt)
    dn = ak.flatten(correct_jets(ev, smear=False, variation="jec_down").Jet.pt)
    assert ak.all(up > nom) and ak.all(dn < nom)


def test_smearing_deterministic_and_sorted():
    from suep_plot.jme import correct_jets

    ev = make_events()
    a = correct_jets(ev)
    b = correct_jets(ev)
    assert ak.all(ak.flatten(a.Jet.pt) == ak.flatten(b.Jet.pt))
    assert ak.all(a.Jet.pt[:, :-1] >= a.Jet.pt[:, 1:])
    assert ak.all(ak.flatten(a.Jet.pt) > 0)


def test_matched_smearing_identity_and_formula():
    from suep_plot.jme import _load_cset

    smear = _load_cset("auto:JME/jer_smear.json.gz")["JERSmear"]
    # matched scaling: factor = 1 + (sf-1)*(pt-genpt)/pt
    assert smear.evaluate(50.0, 0.5, 50.0, 20.0, 11, 0.1, 1.2) == pytest.approx(1.0)
    assert smear.evaluate(50.0, 0.5, 40.0, 20.0, 11, 0.1, 1.2) == pytest.approx(
        1.0 + 0.2 * 10.0 / 50.0)


def test_unmatched_jet_uses_stochastic_seeded_by_event():
    from suep_plot.jme import _load_cset

    smear = _load_cset("auto:JME/jer_smear.json.gz")["JERSmear"]
    # GenPt = -1 -> stochastic; same EventID reproduces, different one varies
    a = smear.evaluate(50.0, 0.5, -1.0, 20.0, 11, 0.1, 1.2)
    b = smear.evaluate(50.0, 0.5, -1.0, 20.0, 11, 0.1, 1.2)
    c = smear.evaluate(50.0, 0.5, -1.0, 20.0, 999, 0.1, 1.2)
    assert a == b and a != c


def test_met_propagation_momentum_balance():
    from suep_plot.jme import correct_jets

    ev = make_events()
    out = correct_jets(ev, smear=False)
    for i in range(2):
        # pair old/new jet pT by phi (unique per jet, unchanged by sorting)
        old_pt = {float(p): float(pt) for p, pt in zip(ev.Jet.phi[i], ev.Jet.pt[i])}
        dx = dy = 0.0
        for j in range(len(out.Jet[i])):
            phi = float(out.Jet.phi[i][j])
            pt_new = float(out.Jet.pt[i][j])
            emf = float(out.Jet.chEmEF[i][j]) + float(out.Jet.neEmEF[i][j])
            if pt_new > 15.0 and emf < 0.9:      # Type-1 delta jet selection
                dx += (pt_new - old_pt[phi]) * np.cos(phi)
                dy += (pt_new - old_pt[phi]) * np.sin(phi)
        mex = float(ev.PuppiMET.pt[i]) * np.cos(float(ev.PuppiMET.phi[i])) - dx
        mey = float(ev.PuppiMET.pt[i]) * np.sin(float(ev.PuppiMET.phi[i])) - dy
        assert float(out.PuppiMET.pt[i]) == pytest.approx(np.hypot(mex, mey))
        assert float(out.PuppiMET.phi[i]) == pytest.approx(np.arctan2(mey, mex))


def test_met_propagation_can_be_disabled():
    from suep_plot.jme import correct_jets

    ev = make_events()
    out = correct_jets(ev, smear=False, met=None)
    assert ak.all(out.PuppiMET.pt == ev.PuppiMET.pt)
    # and MET shifts opposite in sign between jec variations
    up = correct_jets(ev, smear=False, variation="jec_up")
    dn = correct_jets(ev, smear=False, variation="jec_down")
    assert not ak.all(up.PuppiMET.pt == dn.PuppiMET.pt)


def test_bad_variation_raises():
    from suep_plot.jme import correct_jets

    with pytest.raises(ValueError, match="variation"):
        correct_jets(make_events(), variation="banana")
