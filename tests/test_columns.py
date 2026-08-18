"""custom/columns.py configuration: parameters and optional steps."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

columns = pytest.importorskip("custom.columns")

from suep_plot.processor import load_columns_config  # noqa: E402


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
    params, _ = columns.configure({"parameters": {"cluster_eps": 0.2,
                                                  "cluster_min_samples": 50,
                                                  "dr_quantiles": [0.5, 0.68]}})
    assert params["cluster_eps"] == 0.2
    assert columns._dbscan_params("csc") == (0.2, 50)
    assert columns._dbscan_params("rpc") == (0.2, params["rpc_min_samples"])
    # dr_quantiles renames the per-LLP cone fields
    assert columns._dr_field_names() == ("drPairMin", "drPairMax", "drMax",
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
    ({"parameters": {"cluster_eps": -1}}, "cluster_eps must be positive"),
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
        columns.configure(yaml.safe_load(path.read_text()) or {})


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


def test_selected_muons_apply_pt_eta_and_id():
    ak = pytest.importorskip("awkward")
    muons = _muons((25.0, 0.1, 0.0, True),    # keep
                   (25.0, 0.1, 0.5, False),   # fails looseId
                   (5.0, 0.1, 1.0, True),     # fails pt
                   (25.0, 2.9, 1.5, True))    # fails |eta|
    events = ak.zip({"Muon": muons}, depth_limit=1)

    kept = columns._selected_muons(events)

    assert ak.to_list(kept.phi) == [[0.0]]


def test_selected_muons_id_none_keeps_failing_id():
    ak = pytest.importorskip("awkward")
    columns.configure({"parameters": {"iso_muon_id": "none"}})
    events = ak.zip({"Muon": _muons((25.0, 0.1, 0.0, False))}, depth_limit=1)

    assert ak.to_list(columns._selected_muons(events).phi) == [[0.0]]


def test_isolation_measures_dr_only_against_selected_objects():
    """A cluster next to a muon that fails the ID is isolated, not vetoed."""
    ak = pytest.importorskip("awkward")
    clusters = ak.Array([[{"eta": 0.0, "phi": 0.0}]])

    class Ctx:
        pass

    def run(passes_id):
        ctx = Ctx()
        ctx.events = ak.zip({"Muon": _muons((25.0, 0.05, 0.0, passes_id))},
                            depth_limit=1)
        ctx.clusters = {"csc": clusters}
        columns._step_cluster_isolation(ctx)
        return float(ak.flatten(ctx.clusters["csc"].drMuon)[0])

    assert run(True) == pytest.approx(0.05)                      # vetoed
    assert run(False) == pytest.approx(columns.NO_OBJECT_DR)     # isolated


def _jerc_call(monkeypatch, events):
    """Run the jerc step with correct_jets stubbed; return the kwargs it got."""
    import suep_plot.jme as jme

    seen = {}

    def fake_correct_jets(ev, **kwargs):
        seen.update(kwargs)
        return ev

    monkeypatch.setattr(jme, "correct_jets", fake_correct_jets)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.events = events
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
    columns.configure({"parameters": {"jerc": False}})
    events = ak.zip({"Jet": ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])},
                    depth_limit=1)

    assert _jerc_call(monkeypatch, events) == {}


def test_data_without_a_known_data_tag_is_fatal(monkeypatch):
    """Silently skipping the correction would be worse than failing."""
    ak = pytest.importorskip("awkward")
    columns.configure({"parameters": {"jerc_era": "2099_Nonesuch"}})
    events = ak.zip({"Jet": ak.Array([[{"pt": 50.0, "eta": 0.5, "phi": 0.0}]])},
                    depth_limit=1)

    with pytest.raises(ValueError, match="jec_tag_data"):
        _jerc_call(monkeypatch, events)


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
              "match_min_hits", "jerc", "jerc_era", "jerc_algo",
              "iso_jet_pt", "iso_jet_abseta", "iso_jet_id",
              "iso_muon_pt", "iso_muon_abseta", "iso_muon_id")

    grid, _ = _params("configs_mds_grid")
    data, _ = _params("configs_mds_data")

    for key in shared:
        assert grid[key] == data[key], f"{key} differs: {grid[key]} vs {data[key]}"


def test_the_reference_set_agrees_with_the_grid():
    """configs_mds is the reco-only reference for the same clusters."""
    shared = ("cluster_eps", "cluster_min_samples", "rpc_min_samples",
              "match_min_hits", "iso_jet_pt", "iso_jet_id",
              "iso_muon_pt", "iso_muon_id")

    reference, _ = _params("configs_mds")
    grid, _ = _params("configs_mds_grid")

    for key in shared:
        assert reference[key] == grid[key], key


def test_data_set_runs_no_truth_steps():
    """Its histograms touch no gen information, so the llp steps are dropped."""
    _, steps = _params("configs_mds_data")

    assert "clusters" in steps and "cluster_isolation" in steps
    assert "jerc" in steps
    assert not [s for s in steps if s.startswith("llp")]
