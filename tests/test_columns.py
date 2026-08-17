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
    found = sorted(repo.glob("configs*/columns.yaml"))
    for path in found:
        columns.configure(yaml.safe_load(path.read_text()) or {})
