"""custom/columns.py configuration: parameters and optional steps."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

columns = pytest.importorskip("custom.columns")

from suep_plot.processor import load_columns_config  # noqa: E402

# The settings most tests run on; configure() returns them, nothing installs
# them, so a test that wants different ones just makes its own.
DEFAULTS = columns.configure()


@pytest.fixture(autouse=True)
def _restore_defaults():
    """Each test starts (and leaves) the module at its defaults."""
    columns.configure()
    yield
    columns.configure()


def test_defaults():
    params, steps = columns.configure()
    assert params == columns.DEFAULT_PARAMS
    assert steps == tuple(columns.STEP_DEPS)


def test_parameters_override_and_reach_the_helpers():
    settings = columns.configure({"parameters": {"cluster_eps": 0.2,
                                                 "cluster_min_samples": 50,
                                                 "dr_quantiles": [0.5, 0.68]}})
    params = settings.params
    assert params["cluster_eps"] == 0.2
    assert columns._dbscan_params("csc", settings) == (0.2, 50)
    assert columns._dbscan_params("rpc", settings) == (0.2, params["rpc_min_samples"])
    # dr_quantiles renames the per-LLP cone fields
    assert columns._dr_field_names(settings) == ("drPairMin", "drPairMax", "drMax",
                                        "dr50", "dr68")


def test_steps_are_reordered_canonically():
    _, steps = columns.configure({"steps": ["llp_shape", "llp"]})
    assert steps == ("llp", "llp_shape")


@pytest.mark.parametrize("cfg, msg", [
    ({"stpes": []}, "unknown key"),
    ({"parameters": {"eps": 0.2}}, "unknown"),
    ({"parameters": {"llpidx_convention": "nope"}}, "llpidx_convention"),
    ({"steps": ["clusters", "typo"]}, "unknown"),
    ({"steps": ["llp", "llp_reco"]}, "needs"),
    ({"parameters": {"cluster_min_samples": 0}}, "positive integer"),
    ({"parameters": {"cluster_min_samples": 10.5}}, "positive integer"),
    ({"parameters": {"cluster_eps": -1}}, "cluster_eps must be a positive number"),
    ({"parameters": {"dr_quantiles": 0.5}}, "dr_quantiles"),
    ({"parameters": {"dr_quantiles": [0.5, 90]}}, "dr_quantiles"),
])
def test_bad_config_raises(cfg, msg):
    with pytest.raises(ValueError, match=msg):
        columns.configure(cfg)


def test_the_environment_is_ignored(monkeypatch):
    """The settings live in the config set only -- no MDS_* escape hatch."""
    for env, value in (("MDS_CLUSTER_MIN_SAMPLES", "50"),
                       ("MDS_CLUSTER_EPS", "0.2"),
                       ("MDS_SKIP_CLUSTERING", "1"),
                       ("MDS_LLPIDX_CONVENTION", "ordinal")):
        monkeypatch.setenv(env, value)
    params, steps = columns.configure()
    assert params == columns.DEFAULT_PARAMS
    assert steps == tuple(columns.STEP_DEPS)


def test_partial_parameters_keep_the_other_defaults():
    params, _ = columns.configure({"parameters": {"cluster_eps": 0.2}})
    assert params["cluster_eps"] == 0.2
    assert params["cluster_min_samples"] == columns.DEFAULT_PARAMS["cluster_min_samples"]


def test_load_columns_config(tmp_path):
    assert load_columns_config(tmp_path / "columns.yaml") == {}
    path = tmp_path / "columns.yaml"
    path.write_text("steps: [llp]\n")
    assert load_columns_config(path) == {"steps": ["llp"]}


def test_shipped_configs_are_valid():
    """Every columns.yaml in the repo must load and configure cleanly."""
    repo = Path(__file__).resolve().parent.parent
    found = sorted(repo.glob("configs/*/columns.yaml"))
    assert found, "no columns.yaml found -- has the config layout moved again?"
    for path in found:
        # Through the real loader, so _extends is resolved the way a run does it.
        columns.configure(load_columns_config(path))


def test_dr_to_nearest_uses_the_sentinel_when_the_event_has_no_object():
    """A cluster in an event with no muon/jet is maximally isolated, not NaN.

    The isolation selections compare with a plain ``>= 0.4``, which would be
    False on NaN and would drop exactly these clusters.
    """
    ak = pytest.importorskip("awkward")

    clusters = ak.Array([
        [{"eta": 0.0, "phi": 0.0}],          # one muon nearby
        [{"eta": 0.0, "phi": 0.0}],          # no muons in this event
    ])
    muons = ak.Array([[{"eta": 0.1, "phi": 0.0}], []])

    dr = columns._dr_to_nearest(clusters, muons)

    assert dr[0][0] == pytest.approx(0.1)
    assert dr[1][0] == pytest.approx(columns.NO_OBJECT_DR)
    assert columns.NO_OBJECT_DR >= 0.4          # passes any isolation threshold


# ── Cluster isolation: which prompt objects count ──────────────────

def _muons(*specs):
    """specs: (pt, eta, phi, looseId) tuples."""
    ak = pytest.importorskip("awkward")
    return ak.Array([[{"pt": pt, "eta": eta, "phi": phi, "looseId": lid,
                       "mediumId": lid, "tightId": lid}
                      for pt, eta, phi, lid in specs]])


def _ctx(events, settings=None):
    """A real pipeline context on the settings configure() last installed.

    The steps read their parameters off the context, so a test that drives one
    directly has to hand it the same thing derive() would.
    """
    return columns._Context(events, settings or DEFAULTS)


MUON_SEL = {"collection": "Muon",
            "expression": "(obj.pt > 10) & (abs(obj.eta) < 2.4) & obj.looseId"}
JET_SEL = {"collection": "Jet",
           "expression": ("(obj.pt > 20) & (abs(obj.eta) < 2.4)"
                          " & (obj.neHEF < 0.8) & (obj.chHEF > 0.1)"
                          " & jet_id(obj, 'tightlepveto')")}


def test_selected_objects_applies_an_arbitrary_expression():
    ak = pytest.importorskip("awkward")
    muons = _muons((25.0, 0.1, 0.0, True),    # keep
                   (25.0, 0.1, 0.5, False),   # fails looseId
                   (5.0, 0.1, 1.0, True),     # fails pt
                   (25.0, 2.9, 1.5, True))    # fails |eta|
    events = ak.zip({"Muon": muons}, depth_limit=1)

    kept = columns._selected_objects(events, MUON_SEL, DEFAULTS)

    assert ak.to_list(kept.phi) == [[0.0]]


def test_selected_objects_needs_no_code_for_a_new_collection():
    """The whole point: an object type the module has never heard of."""
    ak = pytest.importorskip("awkward")
    electrons = ak.Array([[{"pt": 30.0, "eta": 0.2, "phi": 0.4, "cutBased": 4},
                           {"pt": 30.0, "eta": 0.2, "phi": 0.9, "cutBased": 1}]])
    events = ak.zip({"Electron": electrons}, depth_limit=1)

    kept = columns._selected_objects(events, {
        "collection": "Electron",
        "expression": "(obj.pt > 15) & (obj.cutBased >= 3)"}, DEFAULTS)

    assert ak.to_list(kept.phi) == [[0.4]]


def _jets(*specs):
    """specs: (pt, eta, neHEF, chHEF) tuples; the rest passes the tight ID."""
    ak = pytest.importorskip("awkward")
    return ak.Array([[{"pt": pt, "eta": eta, "phi": 0.1 * i,
                       "chHEF": chhef, "neHEF": nehef, "chEmEF": 0.05,
                       "neEmEF": 0.05, "muEF": 0.0, "chMultiplicity": 10,
                       "neMultiplicity": 5, "nConstituents": 15}
                      for i, (pt, eta, nehef, chhef) in enumerate(specs)]])



def test_selected_objects_exposes_jet_id_to_the_expression():
    ak = pytest.importorskip("awkward")
    pytest.importorskip("correctionlib")

    jets = _jets((50.0, 0.5, 0.30, 0.50),   # keep
                 (50.0, 0.5, 0.90, 0.50),   # fails neHEF < 0.8
                 (50.0, 0.5, 0.30, 0.05),   # fails chHEF > 0.1
                 (10.0, 0.5, 0.30, 0.50),   # fails pt
                 (50.0, 3.5, 0.30, 0.50))   # fails |eta|
    events = ak.zip({"Jet": jets}, depth_limit=1)

    try:
        kept = columns._selected_objects(events, JET_SEL, DEFAULTS)
    except FileNotFoundError:
        pytest.skip("JME payloads not available")

    assert ak.to_list(kept.phi) == [[0.0]]


def test_selected_objects_returns_none_for_a_missing_collection():
    ak = pytest.importorskip("awkward")
    events = ak.zip({"Muon": _muons((25.0, 0.1, 0.0, True))}, depth_limit=1)

    assert columns._selected_objects(events, JET_SEL, DEFAULTS) is None


def test_a_broken_expression_names_the_collection_and_the_expression():
    ak = pytest.importorskip("awkward")
    events = ak.zip({"Muon": _muons((25.0, 0.1, 0.0, True))}, depth_limit=1)

    with pytest.raises(ValueError, match="iso_objects expression for 'Muon'"):
        columns._selected_objects(events, {"collection": "Muon",
                                           "expression": "obj.noSuchField > 1"},
                                  DEFAULTS)


def test_isolation_fields_come_from_the_config():
    """One dR field per iso_objects entry -- including a new one."""
    ak = pytest.importorskip("awkward")
    settings = columns.configure({"parameters": {"iso_objects": {
        "drMuon": MUON_SEL,
        "drSoftMuon": {"collection": "Muon",
                       "expression": "obj.pt > 1"},
    }}})

    ctx = _ctx(ak.zip({"Muon": _muons((25.0, 0.05, 0.0, True))}, depth_limit=1),
               settings)
    ctx.clusters = {"csc": ak.Array([[{"eta": 0.0, "phi": 0.0}]])}
    columns._step_cluster_isolation(ctx)

    assert "drMuon" in ctx.clusters["csc"].fields
    assert "drSoftMuon" in ctx.clusters["csc"].fields
    assert "drJet" not in ctx.clusters["csc"].fields


def test_isolation_measures_dr_only_against_selected_objects():
    """A cluster next to a muon that fails the ID is isolated, not vetoed."""
    ak = pytest.importorskip("awkward")

    settings = columns.configure(
        {"parameters": {"iso_objects": {"drMuon": MUON_SEL}}})

    def run(passes_id):
        ctx = _ctx(ak.zip({"Muon": _muons((25.0, 0.05, 0.0, passes_id))},
                          depth_limit=1), settings)
        ctx.clusters = {"csc": ak.Array([[{"eta": 0.0, "phi": 0.0}]])}
        columns._step_cluster_isolation(ctx)
        return float(ak.flatten(ctx.clusters["csc"].drMuon)[0])

    assert run(True) == pytest.approx(0.05)                      # vetoed
    assert run(False) == pytest.approx(columns.NO_OBJECT_DR)     # isolated


def _jerc_call(monkeypatch, events, settings=None):
    """Run the jerc step with correct_jets stubbed; return the kwargs it got."""
    import suep_plot.jme as jme

    seen = {}

    def fake_correct_jets(ev, **kwargs):
        seen.update(kwargs)
        return ev

    monkeypatch.setattr(jme, "correct_jets", fake_correct_jets)

    ctx = _ctx(events, settings)
    columns._step_jerc(ctx)
    return seen


def test_data_gets_the_residual_corrected_data_tag_and_no_smearing(monkeypatch):
    """L1L2L3Res is the data correction; only JER smearing is MC-only."""
    ak = pytest.importorskip("awkward")
    jets = ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])
    events = ak.zip({"Jet": jets}, depth_limit=1)          # no genWeight = data

    seen = _jerc_call(monkeypatch, events)

    import suep_plot.jme as jme
    assert seen["jec_tag"] == jme.DEFAULTS["2024_Summer24"]["jec_tag_data"]
    assert seen["smear"] is False


def test_mc_gets_the_default_mc_tag_and_smearing(monkeypatch):
    ak = pytest.importorskip("awkward")
    jets = ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])
    events = ak.zip({"Jet": jets, "genWeight": ak.Array([1.0])}, depth_limit=1)

    seen = _jerc_call(monkeypatch, events)

    assert "jec_tag" not in seen        # falls through to the era default (MC)
    assert "smear" not in seen          # correct_jets smears by default


def test_jerc_off_leaves_jets_alone(monkeypatch):
    ak = pytest.importorskip("awkward")
    events = ak.zip({"Jet": ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])},
                    depth_limit=1)

    assert _jerc_call(monkeypatch, events, DEFAULTS.replace(jerc=False)) == {}


def test_data_without_a_known_data_tag_is_fatal(monkeypatch):
    """Silently skipping the correction would be worse than failing."""
    ak = pytest.importorskip("awkward")
    events = ak.zip({"Jet": ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])},
                    depth_limit=1)

    with pytest.raises(ValueError, match="jec_tag_data"):
        _jerc_call(monkeypatch, events, DEFAULTS.replace(jerc_era="2099_Nonesuch"))


def test_jet_id_uses_the_official_payload():
    """2024 NanoAOD has no Jet_jetId; the ID comes from jetid.json.gz."""
    ak = pytest.importorskip("awkward")
    pytest.importorskip("correctionlib")

    good = {"pt": 50.0, "eta": 0.5, "phi": 0.0, "chHEF": 0.5, "neHEF": 0.3,
            "chEmEF": 0.1, "neEmEF": 0.1, "muEF": 0.0, "chMultiplicity": 10,
            "neMultiplicity": 5, "nConstituents": 15}
    noise = {**good, "phi": 1.0, "chHEF": 0.0, "neHEF": 1.0, "chEmEF": 0.0,
             "neEmEF": 0.0, "chMultiplicity": 0, "neMultiplicity": 1,
             "nConstituents": 1}
    jets = ak.Array([[good, noise]])

    try:
        decisions = columns._jet_id(jets, "tight", "2024_Summer24")
    except FileNotFoundError:
        pytest.skip("jsonpog JME payloads not available (no cvmfs)")

    assert ak.to_list(decisions) == [[True, False]]


@pytest.mark.parametrize("cfg, msg", [
    ({"parameters": {"iso_muon_id": "lose"}}, "iso_muon_id"),
    ({"parameters": {"iso_jet_id": "loose"}}, "iso_jet_id"),
    ({"parameters": {"iso_jet_pt": -5}}, "iso_jet_pt"),
    ({"parameters": {"iso_muon_abseta": "2.4"}}, "iso_muon_abseta"),
])
def test_bad_isolation_config_raises(cfg, msg):
    with pytest.raises(ValueError, match=msg):
        columns.configure(cfg)


def _params(cfg):
    repo = Path(__file__).resolve().parent.parent
    params, steps = columns.configure(
        load_columns_config(repo / "configs" / cfg / "columns.yaml"))
    return params, steps


def test_grid_and_data_share_the_clustering_and_isolation_settings():
    """The two sets are overlaid, so these must agree by construction.

    Both pin the values rather than inheriting module defaults: inheriting would
    make the agreement accidental, and a future default change would move one
    study (or both) without anyone noticing.
    """
    shared = ("cluster_eps", "cluster_min_samples", "rpc_min_samples",
              "match_min_hits", "jerc", "jerc_era", "jerc_algo", "iso_objects")

    grid, _ = _params("configs_mds_signal")
    data, _ = _params("configs_mds_data")

    for key in shared:
        assert grid[key] == data[key], f"{key} differs: {grid[key]} vs {data[key]}"


def test_the_reference_set_agrees_with_the_grid():
    """configs_mds is the reco-only reference for the same clusters."""
    shared = ("cluster_eps", "cluster_min_samples", "rpc_min_samples",
              "match_min_hits", "iso_objects")

    reference, _ = _params("configs_mds")
    grid, _ = _params("configs_mds_signal")

    for key in shared:
        assert reference[key] == grid[key], key


def test_data_set_runs_no_truth_steps():
    """Its histograms touch no gen information, so the llp steps are dropped."""
    _, steps = _params("configs_mds_data")

    assert "clusters" in steps and "cluster_isolation" in steps
    assert "jerc" in steps
    assert not [s for s in steps if s.startswith("llp")]


@pytest.mark.parametrize("iso, msg", [
    ("nope", "must be a mapping"),
    ({"drMuon": {"collection": "Muon"}}, "collection' and 'expression'"),
    ({"drMuon": {"collection": "Muon", "expression": "x", "extra": 1}},
     "collection' and 'expression'"),
    ({"drMuon": {"collection": "", "expression": "x"}}, "non-empty string"),
    ({"dr Muon": {"collection": "Muon", "expression": "x"}}, "identifier"),
])
def test_bad_iso_objects_raises(iso, msg):
    with pytest.raises(ValueError, match=msg):
        columns.configure({"parameters": {"iso_objects": iso}})


# ── first-hit chamber: where a cluster starts (punch-through handle) ──────────

def _csc_rechits(hits):
    """One event of CSC-shaped rechits from [(x, y, z, station, chamber)]."""
    import awkward as ak

    x, y, z, station, chamber = (list(v) for v in zip(*hits))
    r = [(xi ** 2 + yi ** 2) ** 0.5 for xi, yi in zip(x, y)]
    import math
    return ak.Array({
        "Eta": [[math.asinh(zi / ri) for zi, ri in zip(z, r)]],
        "Phi": [[math.atan2(yi, xi) for xi, yi in zip(x, y)]],
        "X": [x], "Y": [y], "Z": [z],
        "Station": [station], "Chamber": [chamber],
        "IChamber": [[1] * len(hits)],
        "Tpeak": [[5.0] * len(hits)],
    })


def test_chamber_code_is_the_stored_csc_chamber_unsigned():
    """CSC ships station*10+ring directly; the endcap sign is dropped."""
    rechits = _csc_rechits([(200.0, 0.0, -700.0, -1, -11),
                            (201.0, 0.0, 700.0, 1, 11)])

    codes = columns._flat_chamber_code(rechits)

    assert list(codes) == [11, 11]


def test_chamber_code_is_built_from_station_and_ring_for_dt_and_rpc():
    import awkward as ak

    dt = ak.Array({"SuperLayer": [[1]], "Station": [[2]], "Wheel": [[-1]]})
    rpc_barrel = ak.Array({"Station": [[2]], "Ring": [[1]], "Region": [[0]]})
    rpc_endcap = ak.Array({"Station": [[2]], "Ring": [[1]], "Region": [[1]]})

    assert list(columns._flat_chamber_code(dt)) == [21]
    # the offset is what keeps a barrel and an endcap chamber apart
    assert list(columns._flat_chamber_code(rpc_barrel)) == [21]
    assert list(columns._flat_chamber_code(rpc_endcap)) == [121]


def test_first_hit_is_the_one_closest_to_the_interaction_point():
    """The innermost hit sets firstChamber/firstStation, not the input order."""
    # Deliberately ordered outermost-first: a positional pick would give ME3/1.
    hits = [(200.0, 0.0, 900.0, 3, 31),
            (201.0, 1.0, 700.0, 2, 21),
            (202.0, 2.0, 600.0, 1, 11)]

    clusters = columns._cluster_system(_csc_rechits(hits), None, eps=0.4,
                                       min_samples=3, settings=DEFAULTS)

    assert len(clusters[0]) == 1
    assert clusters[0].firstChamber[0] == 11
    assert clusters[0].firstStation[0] == 1


def test_no_module_holds_settings_of_its_own():
    """The settings travel with the work; no helper reads module state.

    This is what makes a worker process safe: it cannot be "unconfigured",
    because there is nothing to configure -- derive() is handed a Settings and
    passes it down.  A module-level PARAMS is exactly what used to let a worker
    silently cluster with the defaults while the parent printed the config's.
    """
    from custom import clustering, isolation, llp, params

    for mod in (columns, clustering, isolation, llp, params):
        assert not hasattr(mod, "PARAMS"), f"{mod.__name__} kept a global PARAMS"
        assert not hasattr(mod, "STEPS"), f"{mod.__name__} kept a global STEPS"


def test_settings_are_immutable_and_picklable():
    """The processor pickles them to its workers, so they must survive the trip."""
    import pickle

    settings = columns.configure({"parameters": {"cluster_eps": 0.7},
                                  "steps": ["llp", "llp_hits"]})

    assert pickle.loads(pickle.dumps(settings)) == settings
    with pytest.raises(AttributeError):
        settings.params = {}


# ── RPC with the other systems (rpc_mode: merge / match) ─────────────────────

def _rpc_rechits(n, eta, phi, region, times, time_error=1.0, llpidx=-1):
    """One event of RPC-shaped rechits, all in the same chamber."""
    import awkward as ak
    import numpy as np

    ones = np.ones(n)
    return ak.Array({
        "Eta": [list(eta * ones)], "Phi": [list(phi * ones)],
        "X": [list(300.0 * ones)], "Y": [list(100.0 * ones)],
        "Z": [list(800.0 * ones)],
        "Station": [list((2 * ones).astype(int))],
        "Region": [list((region * ones).astype(int))],
        "Ring": [list(ones.astype(int))],
        "Sector": [list((3 * ones).astype(int))],
        "Layer": [list(ones.astype(int))],
        "Time": [list(np.asarray(times, dtype=float))],
        "TimeError": [list(time_error * ones)],
        "Bx": [list(0 * ones)],
        "llpIdx": [list((llpidx * ones).astype(int))],
    })


def test_rpc_mode_must_be_one_of_the_three():
    with pytest.raises(ValueError, match="rpc_mode"):
        columns.configure({"parameters": {"rpc_mode": "yes"}})
    for mode in ("separate", "merge", "match"):
        assert columns.configure({"parameters": {"rpc_mode": mode}})[0]["rpc_mode"] == mode


def test_merged_cluster_absorbs_the_rpc_hits_and_times_them():
    """The RPC hits join the cluster and are what gives it a time.

    The CSC ``time`` (Tpeak) must NOT move when RPC hits are merged in: the two
    detectors are on different clocks, so the RPC estimate is reported apart.
    """
    hits = [(200.0, 0.0, 700.0, 2, 21)] * 6
    csc = _csc_rechits(hits)
    # Same eta-phi as the CSC hits, so DBSCAN puts them in the same cluster.
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(4, eta, phi, region=1, times=[9.0, 10.0, 11.0, 12.0])

    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)
    solo = columns._cluster_system(csc, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)

    assert len(merged[0]) == 1
    assert merged[0].size[0] == 10 and merged[0].nRPCHits[0] == 4
    assert merged[0].rpcHitFrac[0] == pytest.approx(0.4)
    assert merged[0].rpcTime[0] == pytest.approx(10.5)
    assert merged[0].rpcTimeMedian[0] == pytest.approx(10.5)
    # Equal per-hit errors -> the weighted mean is the plain one, sigma/sqrt(N)
    assert merged[0].rpcTimeWeighted[0] == pytest.approx(10.5)
    assert merged[0].rpcTimeErr[0] == pytest.approx(0.5)
    assert merged[0].time[0] == pytest.approx(float(solo[0].time[0]))


def test_an_unfilled_rpc_time_is_nan_and_the_bx_carries_the_timing():
    """MDSNano stores Time = 0 / TimeError = -1: that is no measurement.

    Averaging the placeholder zeros would report every cluster as perfectly
    in time.  The bunch crossing is filled, so it is what dates the cluster.
    """
    import awkward as ak
    import numpy as np

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(4, eta, phi, region=1, times=[0.0] * 4, time_error=-1.0)
    rpc = ak.with_field(rpc, [[0, 0, 1, 2]], "Bx")

    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)

    assert merged[0].nRPCHits[0] == 4
    assert merged[0].rpcTimeValidFrac[0] == 0.0
    for field in ("rpcTime", "rpcTimeMedian", "rpcTimeWeighted", "rpcTimeErr"):
        assert np.isnan(merged[0][field][0])
    assert merged[0].rpcBx[0] == pytest.approx(0.75)
    assert merged[0].rpcBxMedian[0] == pytest.approx(0.5)
    assert merged[0].rpcOutOfTimeFrac[0] == pytest.approx(0.5)


def test_a_merged_cluster_without_rpc_hits_has_no_rpc_time():
    """NaN, not 0: a cluster the RPC never saw is undated, not dated at zero."""
    import numpy as np

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    far = _rpc_rechits(4, -2.0, 3.0, region=1, times=[9.0] * 4)

    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (far, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)

    assert merged[0].nRPCHits[0] == 0
    for field in ("rpcTime", "rpcTimeMedian", "rpcTimeWeighted", "rpcTimeErr"):
        assert np.isnan(merged[0][field][0])


def test_merging_keeps_the_layer_counts_of_the_two_systems_apart():
    """A DT layer id and an RPC layer id collide unless they are offset."""
    import awkward as ak

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(4, eta, phi, region=1, times=[9.0] * 4)

    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)
    solo = columns._cluster_system(csc, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)

    # One CSC layer plus one RPC layer, not one shared layer.
    assert merged[0].nLayer[0] == solo[0].nLayer[0] + 1
    # The innermost hit is still a CSC one, and its chamber code is unchanged.
    assert merged[0].firstSystem[0] == columns.SOURCE_CODES["csc"]
    assert merged[0].firstChamber[0] == solo[0].firstChamber[0]


def test_rpc_regions_split_barrel_from_endcap():
    import awkward as ak

    barrel = _rpc_rechits(2, 0.1, 0.0, region=0, times=[1.0, 2.0])
    endcap = _rpc_rechits(3, 2.0, 0.0, region=1, times=[3.0, 4.0, 5.0])
    rpc = ak.concatenate([barrel, endcap], axis=1)
    events = ak.zip({"rpcRecHits": rpc}, depth_limit=1)

    got_barrel, got_endcap = columns._rpc_regions(events)

    assert list(ak.num(got_barrel.Eta)) == [2]
    assert list(ak.num(got_endcap.Eta)) == [3]


def test_merged_hit_counts_include_the_rpc_region_of_that_system():
    """clusterHitFrac's denominator has to follow what went into the cluster."""
    for mode in ("separate", "match"):
        # "match" leaves the RPC hits outside the cluster, so outside the
        # denominator too -- they are counted by nRPCHits instead.
        settings = columns.configure({"parameters": {"rpc_mode": mode}})
        assert columns._hit_fields("CSC", settings) == ("CSC",)

    merged = columns.configure({"parameters": {"rpc_mode": "merge"}})
    assert columns._hit_fields("CSC", merged) == ("CSC", "RPCEndcap")
    assert columns._hit_fields("DT", merged) == ("DT", "RPCBarrel")


def test_the_processor_carries_the_settings_to_where_derive_runs():
    """The resolved columns.yaml is processor state, so it reaches every worker.

    coffea pickles the processor to its workers; the settings ride along, and
    process() hands them to derive().  There is no second path by which a
    worker could acquire (or miss) them.
    """
    import pickle

    from suep_plot.processor import SuepProcessor

    settings = columns.configure({"parameters": {"rpc_mode": "merge"}})
    proc = SuepProcessor({}, {}, {}, {}, columns.derive, None, settings)

    assert pickle.loads(pickle.dumps(proc)).settings["rpc_mode"] == "merge"


def test_cluster_time_spread_and_oot_fraction_separate_a_shower_from_pile_up():
    """The out-of-time handles: one shower is one time, pile-up is not."""
    import awkward as ak
    # Same geometry twice, so only the hit times differ.
    hits = [(200.0, 0.0, 700.0, 2, 21)] * 6
    shower = _csc_rechits(hits)                      # Tpeak = 5 ns for every hit
    mixed = _csc_rechits(hits)
    mixed = ak.with_field(mixed, [[5.0, 5.0, 5.0, 5.0, 60.0, -40.0]], "Tpeak")

    a = columns._cluster_system(shower, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)
    b = columns._cluster_system(mixed, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)

    assert a[0].timeSpread[0] == pytest.approx(0.0)
    assert a[0].ootHitFrac[0] == pytest.approx(0.0)
    assert b[0].timeSpread[0] > 20
    assert b[0].ootHitFrac[0] == pytest.approx(2 / 6)


def test_the_in_time_window_is_configurable():
    import awkward as ak
    hits = [(200.0, 0.0, 700.0, 2, 21)] * 6
    rechits = _csc_rechits(hits)                     # every hit at 5 ns
    rechits = ak.with_field(rechits, [[5.0] * 6], "Tpeak")

    inside = columns._cluster_system(
        rechits, "Tpeak", eps=0.4, min_samples=3,
        settings=DEFAULTS.replace(oot_time_cut=12.5))
    outside = columns._cluster_system(
        rechits, "Tpeak", eps=0.4, min_samples=3,
        settings=DEFAULTS.replace(oot_time_cut=2.0))

    assert inside[0].ootHitFrac[0] == 0.0
    assert outside[0].ootHitFrac[0] == 1.0


def test_in_time_rpc_hits_are_counted_on_the_merged_cluster():
    """nRPCHitsBx0 makes ">= N in-time RPC hits" expressible on a DT cluster."""
    import awkward as ak
    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(4, eta, phi, region=1, times=[0.0] * 4, time_error=-1.0)
    rpc = ak.with_field(rpc, [[0, 0, 0, 2]], "Bx")

    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)

    assert merged[0].nRPCHits[0] == 4
    assert merged[0].nRPCHitsBx0[0] == 3
    assert merged[0].rpcOutOfTimeFrac[0] == pytest.approx(0.25)


def test_a_rejected_config_installs_nothing():
    """Validation must not leave the bad values behind.

    configure() used to overwrite PARAMS and validate afterwards, so a config
    that raised still installed itself and the next reader picked it up.
    """
    good = columns.configure({"parameters": {"cluster_eps": 0.7}})
    with pytest.raises(ValueError):
        columns.configure({"parameters": {"cluster_eps": 0.9,
                                          "rpc_mode": "nonesuch"}})
    # configure() returns a value instead of installing one, so a rejected
    # config cannot reach anything: the settings in hand are still the good ones.
    assert good["cluster_eps"] == 0.7
    assert good["rpc_mode"] == "separate"


def test_unknown_parameter_name_is_named_on_read():
    with pytest.raises(KeyError, match="no such parameter"):
        columns.configure()["cluster_epsilon"]


def _empty_dt():
    """One event with no DT rechits (DT-shaped fields, so _flat_hits sees DT)."""
    import awkward as ak
    return ak.Array({f: [[]] for f in ("Eta", "Phi", "X", "Y", "Z", "Wheel",
                                       "Sector", "Station", "SuperLayer",
                                       "Layer")})


def test_derive_takes_its_settings_as_arguments():
    """The pipeline runs on what it is handed, not on module state.

    This is what lets the configuration travel with the work instead of having
    to be installed in every process that might run derive().
    """
    ak = pytest.importorskip("awkward")
    settings = columns.configure({"parameters": {"cluster_min_samples": 3},
                                  "steps": ["clusters"]})
    csc = _csc_rechits([(200.0, 0.0, -700.0, 1, 11),
                        (201.0, 1.0, -701.0, 1, 11),
                        (202.0, 2.0, -702.0, 2, 21)])
    events = ak.zip({"cscRechits": csc, "dtRecHits": _empty_dt(),
                     "rpcRecHits": _rpc_rechits(0, 0.0, 0.0, 1, [])},
                    depth_limit=1)

    out = columns.derive(events, settings)
    assert len(ak.flatten(out.cscCluster.size)) == 1
    # min_samples raised past the cluster's size: a second, independent
    # Settings, so the two can be run side by side without interfering.
    stricter = settings.replace(cluster_min_samples=10)
    assert len(ak.flatten(columns.derive(events, stricter).cscCluster.size)) == 0
    assert settings["cluster_min_samples"] == 3


def test_the_rpc_offset_does_not_depend_on_the_component_order():
    """RPC is offset because it is RPC, not because it came second.

    Keying the offset on the position in ``components`` made the shift mean
    "the second collection", which is only unambiguous as long as every merge
    happens to pass RPC last.
    """
    import awkward as ak

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(4, eta, phi, region=1, times=[9.0] * 4)

    def first_chambers(components):
        clusters = columns._cluster_merged(components, eps=0.4, min_samples=3,
                                     settings=DEFAULTS)
        return sorted(float(v) for v in ak.flatten(clusters.firstChamber))

    csc_first = first_chambers([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")])
    rpc_first = first_chambers([(rpc, "Time", "rpc"), (csc, "Tpeak", "csc")])
    assert csc_first == rpc_first
    # ... and it is the CSC code, unshifted, either way.
    solo = columns._cluster_system(csc, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)
    assert csc_first == [float(solo[0].firstChamber[0])]


def test_merging_two_non_rpc_systems_is_refused():
    """Two systems sharing offset 0 would collapse different chambers into one id."""
    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 6)
    with pytest.raises(ValueError, match="at most one non-RPC"):
        columns._cluster_merged([(csc, "Tpeak", "csc"), (_empty_dt(), None, "dt")],
                                eps=0.4, min_samples=3, settings=DEFAULTS)


# ── rpc_mode: match — RPC dates the clusters without joining them ─────────────

def _matched(csc, rpc, eps=0.4, min_samples=3):
    """CSC clusters of *csc*, annotated with the RPC hits of *rpc*."""
    clusters = columns._cluster_system(csc, "Tpeak", eps=eps, min_samples=min_samples,
                        settings=DEFAULTS)
    return columns._match_rpc(clusters, rpc, eps, "csc")


def test_matching_does_not_change_the_clustering():
    """The whole point: RPC cannot make, unmake or move a cluster.

    Merging lets RPC hits count towards min_samples, which in data -- where
    only ~19% of RPC rechits are at BX 0 -- means noise can push a background
    cluster over threshold.
    """
    import awkward as ak

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 4)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(20, eta, phi, region=1, times=[9.0] * 20)

    solo = columns._cluster_system(csc, "Tpeak", eps=0.4, min_samples=3,
                        settings=DEFAULTS)
    matched = _matched(csc, rpc)
    merged = columns._cluster_merged([(csc, "Tpeak", "csc"), (rpc, "Time", "rpc")],
                                     eps=0.4, min_samples=3,
                                     settings=DEFAULTS)
    for field in ("size", "eta", "phi", "nLayer", "firstChamber", "time"):
        assert ak.to_list(matched[field]) == ak.to_list(solo[field]), field
    # ... whereas merging absorbs them, so the cluster is a different object.
    assert merged[0].size[0] == solo[0].size[0] + 20


def test_matching_still_reports_the_rpc_content_and_timing():
    import awkward as ak

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 4)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    rpc = _rpc_rechits(6, eta, phi, region=1, times=[9.0] * 6)

    clusters = _matched(csc, rpc)
    assert clusters[0].nRPCHits[0] == 6
    assert clusters[0].nRPCHitsBx0[0] == 6
    assert clusters[0].rpcBx[0] == 0
    # rpcHitFrac means the same thing as under merge: the RPC share of the
    # cluster's hits, 6 / (4 + 6).
    assert clusters[0].rpcHitFrac[0] == pytest.approx(0.6)
    # No RPC hit is in the cluster, so it can never be the innermost one.
    assert clusters[0].firstSystem[0] == columns.SOURCE_CODES["csc"]


def test_matching_ignores_rpc_hits_beyond_eps():
    import numpy as np

    csc = _csc_rechits([(200.0, 0.0, 700.0, 2, 21)] * 4)
    eta, phi = float(csc.Eta[0][0]), float(csc.Phi[0][0])
    far = _rpc_rechits(5, eta + 2.0, phi, region=1, times=[9.0] * 5)

    clusters = _matched(csc, far)
    assert clusters[0].nRPCHits[0] == 0
    assert clusters[0].rpcHitFrac[0] == 0.0
    assert np.isnan(clusters[0].rpcBx[0])


def test_an_rpc_hit_dates_only_its_nearest_cluster():
    """Otherwise a hit between two clusters would be counted twice."""
    import awkward as ak

    hits = [(200.0, 0.0, 700.0, 2, 21)] * 4 + [(200.0, 0.0, 300.0, 1, 11)] * 4
    csc = _csc_rechits(hits)
    clusters = columns._cluster_system(csc, "Tpeak", eps=0.05, min_samples=3,
                        settings=DEFAULTS)
    assert len(clusters[0].eta) == 2                      # two separate clusters
    eta = float(ak.flatten(clusters.eta)[0])
    phi = float(ak.flatten(clusters.phi)[0])
    rpc = _rpc_rechits(3, eta, phi, region=1, times=[9.0] * 3)

    got = columns._match_rpc(clusters, rpc, 10.0, "csc")   # eps large enough for both
    assert sorted(ak.to_list(ak.flatten(got.nRPCHits))) == [0, 3]
