"""Reweighter and map-generation tests (no ROOT files needed)."""

import pickle

import awkward as ak
import numpy as np
import pytest

from suep_plot.reweights import (
    Reweighter,
    _binned_lookup,
    _binned_lookup_2d,
    build_reweighters,
    make_reweight_map,
)


def make_events():
    jets = ak.Array([
        [{"pt": 50.0, "eta": 1.0}, {"pt": 100.0, "eta": -2.0}],
        [],
        [{"pt": 200.0, "eta": 0.5}],
    ])
    met = ak.Array([10.0, 60.0, 120.0])
    return ak.zip({"Jet": jets, "met": met}, depth_limit=1)


# ── binned lookups ────────────────────────────────────────────────


def test_binned_lookup_inside_outside_nan():
    edges = np.array([0.0, 10.0, 20.0])
    weights = np.array([2.0, 3.0])
    vals = np.array([5.0, 15.0, 25.0, -1.0, np.nan])
    out = _binned_lookup(vals, edges, weights, clamp=False)
    assert list(out) == [2.0, 3.0, 1.0, 1.0, 1.0]
    out = _binned_lookup(vals, edges, weights, clamp=True)
    assert list(out) == [2.0, 3.0, 3.0, 2.0, 1.0]


def test_binned_lookup_2d():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    ex, ey = np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 20.0])
    out = _binned_lookup_2d(np.array([0.5, 1.5, 5.0]), np.array([5.0, 15.0, 5.0]),
                            ex, ey, w, clamp=False)
    assert list(out) == [1.0, 4.0, 1.0]


# ── Reweighter ────────────────────────────────────────────────────


def test_expression_event_weight():
    rw = Reweighter("r", {"expression": "events.met / 10.0"})
    assert list(rw.event_weight(make_events())) == [1.0, 6.0, 12.0]


def test_object_level_product_and_empty_events():
    rw = Reweighter("r", {"expression": "events.Jet.pt / 100.0", "level": "object"})
    w = rw.event_weight(make_events())
    assert w == pytest.approx([0.5, 1.0, 2.0])  # empty event -> 1


def test_binned_object_level():
    rw = Reweighter("r", {"variable": "events.Jet.pt", "edges": [0, 100, 400],
                          "weights": [0.5, 2.0], "level": "object"})
    w = rw.event_weight(make_events())
    assert w == pytest.approx([0.5 * 2.0, 1.0, 2.0])


def test_applies_to_rules():
    rw = Reweighter("r", {"expression": "1.0", "apply_to": ["background"]})
    assert rw.applies_to("qcd", "background", is_data=False)
    assert not rw.applies_to("sig", "signal", is_data=False)
    assert not rw.applies_to("data", "", is_data=True)

    rw = Reweighter("r", {"expression": "1.0", "samples": ["data"]})
    assert rw.applies_to("data", "", is_data=True)
    assert not rw.applies_to("qcd", "background", is_data=False)

    rw = Reweighter("r", {"expression": "1.0", "fill_only": True})
    assert not rw.applies_to("qcd", "background", is_data=False)


def test_config_validation():
    with pytest.raises(ValueError, match="len\\(weights\\)"):
        Reweighter("r", {"variable": "events.met", "edges": [0, 1, 2], "weights": [1.0]})
    with pytest.raises(ValueError, match="level"):
        Reweighter("r", {"expression": "1.0", "level": "banana"})
    with pytest.raises(ValueError, match="need"):
        Reweighter("r", {"apply_to": "all"})


def test_file_include_with_override(tmp_path):
    (tmp_path / "map.yaml").write_text(
        "variable: 'events.met'\nedges: [0, 100]\nweights: [2.0]\napply_to: [signal]\n")
    rws = build_reweighters(
        {"m": {"file": "map.yaml", "apply_to": ["background"]}}, tmp_path)
    assert rws["m"].apply_to == ["background"]  # entry overrides file
    assert list(rws["m"].event_weight(make_events())) == [2.0, 2.0, 1.0]


# ── suep-reweight map generation ──────────────────────────────────


def test_make_reweight_map_roundtrip(tmp_path):
    import hist

    h = hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.Regular(2, 0, 100, name="x"),
        storage=hist.storage.Weight(),
    )
    h.fill(dataset="num", x=[25] * 30 + [75] * 10)
    h.fill(dataset="den", x=[25] * 20 + [75] * 20)

    payload = {"histograms": {"met": h},
               "hist_defs": {"met": {"expression": "events.met",
                                     "bins": 2, "lo": 0, "hi": 100}},
               "samples": {}, "sumw": {}, "nevents": {}}
    pkl = tmp_path / "x.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)

    out = tmp_path / "map.yaml"
    make_reweight_map([str(pkl)], "met", "num", "den", str(out), normalize=True)

    rws = build_reweighters({"m": {"file": "map.yaml"}}, tmp_path)
    # shape ratio: (0.75/0.5, 0.25/0.5) = (1.5, 0.5)
    w = rws["m"].event_weight(make_events())  # met = 10, 60, 120 (120 clamps)
    assert w == pytest.approx([1.5, 0.5, 0.5])
