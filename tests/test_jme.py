"""JEC/JER tests on synthetic jets (skipped when jsonpog files are absent)."""

import functools
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

ERA = "2024_Summer24"


def _payload_available():
    """True when a jet_jerc payload resolves for the era these tests use."""
    try:
        from suep_plot.jme import payload_path

        return Path(payload_path(ERA, "jet_jerc.json.gz")).is_file()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _payload_available(),
    reason="no JME jet_jerc payload reachable (CAT metadata / jsonpog / "
           "$CORRECTIONLIB_DATA)",
)


def make_events(gen_pt=None, low_pt_jets=False, musub_dphi=True):
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
        "muonSubtrFactor": jag([0.0, 0.0, 0.0, 0.0]),
        "muonSubtrDeltaPhi": jag([0.0, 0.0, 0.0, 0.0]),
    })
    if not musub_dphi:
        # field selection, not ak.zip: it keeps the jagged layout intact
        jets = jets[[f for f in jets.fields if f != "muonSubtrDeltaPhi"]]
    if gen_pt is None:
        gen_pt = jag([50.0, 30.0, 80.0, 20.0])
    genjets = ak.zip({
        "pt": gen_pt,
        "eta": jag([0.5, -1.2, 1.8, 0.1]),
        "phi": jag([0.1, 2.0, -1.0, 3.0]),
    })
    rho = ak.zip({"fixedGridRhoFastjetAll": ak.Array([20.0, 25.0])})
    met = ak.zip({"pt": ak.Array([40.0, 60.0]), "phi": ak.Array([0.3, -2.0])})
    # The Type-1 correction is rebuilt from the raw MET, so both are needed.
    raw_met = ak.zip({"pt": ak.Array([45.0, 70.0]), "phi": ak.Array([0.2, -2.1])})
    fields = {"Jet": jets, "GenJet": genjets, "Rho": rho, "PuppiMET": met,
              "RawPuppiMET": raw_met, "event": ak.Array([11, 12])}
    if low_pt_jets:
        # CorrT1METJet: the sub-15 GeV jets NanoAOD stores for the T1 rebuild.
        fields["CorrT1METJet"] = ak.zip({
            # 14.x corrects above 15 GeV, so these really enter the sum
            "rawPt": ak.Array([[14.5], [14.9]]),
            "eta": ak.Array([[0.3], [-0.7]]),
            "phi": ak.Array([[1.0], [2.5]]),
            "area": ak.Array([[0.5], [0.5]]),
            "muonSubtrFactor": ak.Array([[0.0], [0.0]]),
            "EmEF": ak.Array([[0.1], [0.1]]),
        })
    return ak.zip(fields, depth_limit=1)


def _tag_kwargs():
    tags = _tags()
    return {"jec_tag": tags["jec_tag"], "jer_tag": tags["jer_tag"]}


@functools.lru_cache(maxsize=None)
def _tags():
    """Payload tags to test against.

    Deliberately taken from the payload rather than from ``jme.DEFAULTS``: the
    defaults name the current JME *recommendation*, which may be newer than the
    jsonpog copy on this machine, and these tests are about the code path, not
    about which calibration is recommended today.
    """
    from suep_plot.jme import _load_cset, payload_path

    algo = "AK4PFPuppi"
    cset = _load_cset(payload_path(ERA, "jet_jerc.json.gz"))
    jec = sorted(k.rsplit("_L1L2L3Res", 1)[0] for k in cset.compound
                 if k.endswith(f"_L1L2L3Res_{algo}") and "_MC_" in k)
    jec_data = sorted(k.rsplit("_L1L2L3Res", 1)[0] for k in cset.compound
                      if k.endswith(f"_L1L2L3Res_{algo}") and "_DATA_" in k)
    jer = sorted(k.rsplit("_PtResolution", 1)[0] for k in cset
                 if k.endswith(f"_PtResolution_{algo}"))
    if not (jec and jec_data and jer):
        pytest.skip("no usable JEC/JER tags in the available payload")
    return {"jec_tag": jec[0], "jer_tag": jer[0], "jec_tag_data": jec_data[0]}


def test_jec_variations_bracket_nominal():
    from suep_plot.jme import correct_jets

    ev = make_events()
    nom = ak.flatten(correct_jets(ev, smear=False, **_tag_kwargs()).Jet.pt)
    up = ak.flatten(correct_jets(ev, smear=False, variation="jec_up", **_tag_kwargs()).Jet.pt)
    dn = ak.flatten(correct_jets(ev, smear=False, variation="jec_down", **_tag_kwargs()).Jet.pt)
    assert ak.all(up > nom) and ak.all(dn < nom)


def test_smearing_deterministic_and_sorted():
    from suep_plot.jme import correct_jets

    ev = make_events()
    a = correct_jets(ev, **_tag_kwargs())
    b = correct_jets(ev, **_tag_kwargs())
    assert ak.all(ak.flatten(a.Jet.pt) == ak.flatten(b.Jet.pt))
    assert ak.all(a.Jet.pt[:, :-1] >= a.Jet.pt[:, 1:])
    assert ak.all(ak.flatten(a.Jet.pt) > 0)


def test_gen_matching_drives_the_hybrid_smearing():
    """A perfectly gen-matched jet is barely smeared; an unmatched one is not.

    The factory does the smearing, but deciding which jets are matched (dR < 0.2
    and within 3 sigma) is still this module's job -- ``_matched_gen_pt`` feeds
    the ``ptGenJet`` column the hybrid method keys on.
    """
    from suep_plot.jme import correct_jets

    matched = correct_jets(make_events(), **_tag_kwargs())          # genJetIdx 0,1 / 0,-1
    # gen pT far from reco pT -> fails the 3-sigma test -> stochastic instead
    unmatched = correct_jets(make_events(gen_pt=ak.Array([[5.0, 3.0], [8.0, 2.0]])), **_tag_kwargs())

    assert ak.all(ak.flatten(matched.Jet.pt) > 0)
    assert ak.all(ak.flatten(unmatched.Jet.pt) > 0)
    # the two must differ: the smearing method actually changed
    assert ak.to_list(ak.flatten(matched.Jet.pt)) != \
        ak.to_list(ak.flatten(unmatched.Jet.pt))


def test_helper_columns_do_not_leak_into_the_output_jets():
    """pt_raw / mass_raw / event_rho / pt_gen are factory plumbing, not physics.

    'event_rho' in particular must not be called 'rho': on a real coffea
    JetArray that name is an azimuthal-radial alias for pt, and adding it breaks
    the vector behaviour of the whole collection.
    """
    from suep_plot.jme import correct_jets

    out = correct_jets(make_events(), **_tag_kwargs())

    for leaked in ("pt_raw", "mass_raw", "event_rho", "rho", "pt_gen"):
        assert leaked not in out.Jet.fields


def test_correct_jets_uses_coffea_s_factory():
    """The JEC/JER algebra is coffea's, not a local reimplementation."""
    import inspect

    from suep_plot import jme

    src = inspect.getsource(jme.correct_jets)
    assert "CorrectedJetsFactory" in src


def test_met_is_rebuilt_from_the_raw_met_not_the_stored_one():
    """Shifting the raw MET shifts the result; the stored MET is only a template.

    This is the whole point of a rebuild rather than a delta: the Type-1 term is
    recomputed from scratch, so the already-corrected value carries no weight.
    """
    from suep_plot.jme import correct_jets

    base = correct_jets(make_events(), smear=False, **_tag_kwargs())

    # move the raw MET by (+10, 0) in x: the output must move by exactly that
    ev = make_events()
    raw = ev.RawPuppiMET
    px = raw.pt * np.cos(raw.phi) + 10.0
    py = raw.pt * np.sin(raw.phi)
    shifted = ak.with_field(raw, np.hypot(px, py), "pt")
    shifted = ak.with_field(shifted, np.arctan2(py, px), "phi")
    ev = ak.with_field(ev, shifted, "RawPuppiMET")
    moved = correct_jets(ev, smear=False, **_tag_kwargs())

    for i in range(2):
        bx = float(base.PuppiMET.pt[i]) * np.cos(float(base.PuppiMET.phi[i]))
        by = float(base.PuppiMET.pt[i]) * np.sin(float(base.PuppiMET.phi[i]))
        mx = float(moved.PuppiMET.pt[i]) * np.cos(float(moved.PuppiMET.phi[i]))
        my = float(moved.PuppiMET.pt[i]) * np.sin(float(moved.PuppiMET.phi[i]))
        assert mx - bx == pytest.approx(10.0)
        assert my - by == pytest.approx(0.0, abs=1e-9)

    # and the stored MET must not enter at all
    ev2 = make_events()
    junk = ak.with_field(ev2.PuppiMET, ev2.PuppiMET.pt * 3.0 + 100.0, "pt")
    ev2 = ak.with_field(ev2, junk, "PuppiMET")
    out2 = correct_jets(ev2, smear=False, **_tag_kwargs())
    assert ak.to_list(out2.PuppiMET.pt) == ak.to_list(base.PuppiMET.pt)


def test_met_rebuild_includes_corrt1metjet():
    """The sub-15 GeV jets NanoAOD stores for the rebuild must be summed too.

    Leaving them out is a silent bias: only the jets whose *corrected* pT crosses
    15 GeV contribute, so a rawPt just under the Jet threshold still counts.
    """
    from suep_plot.jme import correct_jets

    without = correct_jets(make_events(), smear=False, **_tag_kwargs())
    with_low = correct_jets(make_events(low_pt_jets=True), smear=False,
                            **_tag_kwargs())

    assert ak.to_list(with_low.PuppiMET.pt) != ak.to_list(without.PuppiMET.pt), \
        "CorrT1METJet jets crossing 15 GeV did not reach the MET sum"


def test_corrt1metjet_below_threshold_is_ignored():
    """The 15 GeV cut is on the corrected pT, so genuinely soft jets drop out."""
    from suep_plot.jme import correct_jets

    ev = make_events(low_pt_jets=True)
    soft = ak.with_field(ev.CorrT1METJet, ak.Array([[5.0], [6.0]]), "rawPt")
    ev = ak.with_field(ev, soft, "CorrT1METJet")

    assert ak.to_list(correct_jets(ev, smear=False, **_tag_kwargs()).PuppiMET.pt) \
        == ak.to_list(correct_jets(make_events(), smear=False,
                                   **_tag_kwargs()).PuppiMET.pt)


def test_met_rebuild_without_a_raw_counterpart_is_fatal():
    """Silently leaving MET uncorrected would be worse than stopping."""
    from suep_plot.jme import correct_jets

    ev = make_events()
    no_raw = ak.zip({f: ev[f] for f in ev.fields if f != "RawPuppiMET"},
                    depth_limit=1)

    with pytest.raises(ValueError, match="RawPuppiMET"):
        correct_jets(no_raw, smear=False, **_tag_kwargs())


def test_smearing_reaches_the_met_sum(monkeypatch):
    """A JER factor on the jets must move the MET too, not just the jets."""
    from suep_plot.jme import correct_jets

    ev = make_events()
    unsmeared = correct_jets(ev, smear=False, **_tag_kwargs())
    _stub_smear(monkeypatch, [1.5] * 4)
    smeared = correct_jets(ev, **_tag_kwargs())

    assert ak.to_list(smeared.PuppiMET.pt) != ak.to_list(unsmeared.PuppiMET.pt)


def test_met_propagation_can_be_disabled():
    from suep_plot.jme import correct_jets

    ev = make_events()
    out = correct_jets(ev, smear=False, met=None, **_tag_kwargs())
    assert ak.all(out.PuppiMET.pt == ev.PuppiMET.pt)
    # and MET shifts opposite in sign between jec variations
    up = correct_jets(ev, smear=False, variation="jec_up", **_tag_kwargs())
    dn = correct_jets(ev, smear=False, variation="jec_down", **_tag_kwargs())
    assert not ak.all(up.PuppiMET.pt == dn.PuppiMET.pt)


def test_bad_variation_raises():
    from suep_plot.jme import correct_jets

    with pytest.raises(ValueError, match="variation"):
        correct_jets(make_events(), variation="banana", **_tag_kwargs())


def test_data_tag_needs_the_run_input_and_is_supplied():
    """The DATA compound is run-dependent; one payload covers the whole year."""
    from suep_plot.jme import correct_jets

    data_tag = _tags()["jec_tag_data"]
    events = make_events()
    # data-like: no GenJet, and a real run number
    fields = {f: events[f] for f in events.fields if f != "GenJet"}
    fields["run"] = ak.Array([380043, 380043])
    data = ak.zip(fields, depth_limit=1)

    out = correct_jets(data, jec_tag=data_tag, smear=False)

    assert ak.all(out.Jet.pt > 0)
    # eta/phi are untouched by JEC (only pT/mass scale), so the dR geometry holds
    assert sorted(ak.to_list(ak.flatten(out.Jet.eta))) == \
        sorted(ak.to_list(ak.flatten(data.Jet.eta)))


def test_data_tag_without_run_raises_a_named_error():
    from suep_plot.jme import correct_jets

    data_tag = _tags()["jec_tag_data"]
    events = make_events()
    no_run = ak.zip({f: events[f] for f in events.fields if f != "GenJet"},
                    depth_limit=1)

    with pytest.raises(KeyError, match="run"):
        correct_jets(no_run, jec_tag=data_tag, smear=False)


def test_smearing_uses_the_shipped_jersmear_payload():
    """The JERSmear helper is a package data file, not a cvmfs lookup."""
    from pathlib import Path

    from suep_plot.jme import _smear_cset

    shipped = Path(__file__).resolve().parent.parent / "src/suep_plot/jer_smear.json.gz"
    assert shipped.is_file(), "jer_smear.json.gz must ship with the package"
    assert [i.name for i in _smear_cset()["JERSmear"].inputs] == [
        "JetPt", "JetEta", "GenPt", "Rho", "EventID", "JER", "JERSF"]


def test_jersmear_matched_and_stochastic_modes():
    """GenPt >= 0 scales toward the gen jet; -1 switches to stochastic."""
    from suep_plot.jme import _smear_cset

    smear = _smear_cset()["JERSmear"]
    # matched scaling: factor = 1 + (sf-1)*(pt-genpt)/pt
    assert smear.evaluate(50.0, 0.5, 50.0, 20.0, 11, 0.1, 1.2) == pytest.approx(1.0)
    assert smear.evaluate(50.0, 0.5, 40.0, 20.0, 11, 0.1, 1.2) == pytest.approx(
        1.0 + 0.2 * 10.0 / 50.0)
    # GenPt = -1 -> stochastic, seeded by EventID so it reproduces
    a = smear.evaluate(50.0, 0.5, -1.0, 20.0, 11, 0.1, 1.2)
    b = smear.evaluate(50.0, 0.5, -1.0, 20.0, 11, 0.1, 1.2)
    c = smear.evaluate(50.0, 0.5, -1.0, 20.0, 999, 0.1, 1.2)
    assert a == b and a != c


def _stub_smear(monkeypatch, values):
    """Force JERSmear to return *values* (one per jet, flat order)."""
    import numpy as np

    from suep_plot import jme

    class Stub:
        inputs = [type("I", (), {"name": n})
                  for n in ("JetPt", "JetEta", "GenPt", "Rho", "EventID",
                            "JER", "JERSF")]

        def evaluate(self, *args):
            return np.asarray(values, dtype=float)

    monkeypatch.setattr(jme, "_smear_cset", lambda: {"JERSmear": Stub()})


def test_bad_smear_factor_falls_back_to_no_smearing(monkeypatch):
    """Non-finite / non-positive factors mean 1.0, never a killed jet."""
    from suep_plot.jme import correct_jets

    ev = make_events()
    unsmeared = ak.to_list(correct_jets(ev, smear=False, **_tag_kwargs()).Jet.pt)

    for bad in ([float("nan")] * 4, [-0.5] * 4, [0.0] * 4):
        _stub_smear(monkeypatch, bad)
        out = correct_jets(ev, **_tag_kwargs())
        assert ak.to_list(out.Jet.pt) == unsmeared, f"factor {bad[0]} altered the jets"


def test_good_smear_factor_is_applied_to_pt_and_mass(monkeypatch):
    from suep_plot.jme import correct_jets

    ev = make_events()
    base = correct_jets(ev, smear=False, **_tag_kwargs())
    _stub_smear(monkeypatch, [2.0] * 4)
    out = correct_jets(ev, **_tag_kwargs())

    assert ak.to_list(out.Jet.pt) == [[2 * v for v in row]
                                     for row in ak.to_list(base.Jet.pt)]
    assert ak.to_list(out.Jet.mass) == [[2 * v for v in row]
                                       for row in ak.to_list(base.Jet.mass)]


def test_default_tags_exist_in_the_resolved_payload():
    """DEFAULTS must name tags the payload actually has.

    jsonpog's 2024_Summer24 only carries Summer24Prompt24_V1, so before the CAT
    campaign directory was wired in, the configured V5 tags resolved to nothing
    and every run died on a KeyError.
    """
    from suep_plot.jme import DEFAULTS, _load_cset, payload_path

    algo = "AK4PFPuppi"
    cset = _load_cset(payload_path(ERA, "jet_jerc.json.gz"))
    cfg = DEFAULTS[ERA]

    for key in ("jec_tag", "jec_tag_data"):
        assert f"{cfg[key]}_L1L2L3Res_{algo}" in cset.compound, \
            f"{key}={cfg[key]!r} missing from {payload_path(ERA, 'jet_jerc.json.gz')}"
    assert f"{cfg['jer_tag']}_PtResolution_{algo}" in cset
    assert f"{cfg['jer_tag']}_ScaleFactor_{algo}" in cset


def test_payload_comes_from_the_campaign_directory():
    """The campaign dir wins over jsonpog, so JEC and jet ID stay in step."""
    from suep_plot.jme import DEFAULTS, payload_path

    campaign = DEFAULTS[ERA].get("payload_dir")
    if not campaign or not Path(campaign).is_dir():
        pytest.skip("CAT metadata not mounted")

    for filename in ("jet_jerc.json.gz", "jetid.json.gz"):
        assert payload_path(ERA, filename) == str(Path(campaign) / filename)


def test_met_uses_the_muon_subtracted_jet_axis():
    """Removing the muon rotates the jet, so MET projects along phi + dPhi."""
    from suep_plot.jme import correct_jets

    ev = make_events()
    rotated = ak.with_field(ev.Jet, ak.Array([[0.0, 0.4], [0.0, 0.0]]),
                            "muonSubtrDeltaPhi")
    ev = ak.with_field(ev, rotated, "Jet")

    turned = correct_jets(ev, smear=False, **_tag_kwargs())
    straight = correct_jets(make_events(), smear=False, **_tag_kwargs())

    # only the first event has a rotated jet, so only its MET may move
    assert float(turned.PuppiMET.pt[0]) != pytest.approx(
        float(straight.PuppiMET.pt[0]))
    assert float(turned.PuppiMET.pt[1]) == pytest.approx(
        float(straight.PuppiMET.pt[1]))


def test_met_falls_back_to_jet_phi_without_the_delta_branch():
    """Older NanoAOD has no muonSubtrDeltaPhi: use the jet axis as it is."""
    from suep_plot.jme import correct_jets

    assert "muonSubtrDeltaPhi" not in make_events(musub_dphi=False).Jet.fields

    no_branch = correct_jets(make_events(musub_dphi=False), smear=False,
                             **_tag_kwargs())
    zero_delta = correct_jets(make_events(), smear=False, **_tag_kwargs())

    # the fixture's deltas are zero, so the fallback must agree exactly
    assert ak.to_list(no_branch.PuppiMET.pt) == \
        pytest.approx(ak.to_list(zero_delta.PuppiMET.pt))


def test_muon_subtraction_reaches_the_met_but_not_the_jets():
    """The asymmetry is the prescription, not an oversight.

    The JEC was derived for the whole jet, muon content included, so the jets
    handed back use the full pT.  MET already counts the muon directly, so the
    Type-1 sum uses the muon-subtracted jet -- correcting the muon's momentum as
    if it were hadronic would double count it.
    """
    from suep_plot.jme import correct_jets

    plain = correct_jets(make_events(), smear=False, **_tag_kwargs())

    ev = make_events()
    with_muons = ak.with_field(ev.Jet, ak.Array([[0.3, 0.0], [0.0, 0.0]]),
                               "muonSubtrFactor")
    ev = ak.with_field(ev, with_muons, "Jet")
    subtracted = correct_jets(ev, smear=False, **_tag_kwargs())

    # the jets are untouched by muonSubtrFactor ...
    assert ak.to_list(ak.flatten(subtracted.Jet.pt)) == pytest.approx(
        ak.to_list(ak.flatten(plain.Jet.pt)))
    assert ak.to_list(ak.flatten(subtracted.Jet.mass)) == pytest.approx(
        ak.to_list(ak.flatten(plain.Jet.mass)))
    # ... while the MET sum does see it
    assert float(subtracted.PuppiMET.pt[0]) != pytest.approx(
        float(plain.PuppiMET.pt[0]))
