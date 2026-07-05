"""Fill logic tests on synthetic (ROOT-free) awkward events."""

import awkward as ak
import numpy as np
import pytest

from suep_plot.histograms import build_histograms, fill_histograms
from suep_plot.reweights import Reweighter


def make_events():
    """3 events: 2 jets / 0 jets / 1 jet."""
    jets = ak.Array([
        [{"pt": 50.0, "eta": 1.0}, {"pt": 100.0, "eta": -2.0}],
        [],
        [{"pt": 200.0, "eta": 0.5}],
    ])
    met = ak.Array([10.0, 60.0, 120.0])
    return ak.zip({"Jet": jets, "met": met}, depth_limit=1)


def totals(h, sample="s"):
    return h[{"dataset": sample}].sum(flow=True).value


def test_event_level_fill_and_weight():
    events = make_events()
    defs = {"met": {"expression": "events.met", "bins": 10, "lo": 0, "hi": 200}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events, "s", np.array([1.0, 2.0, 3.0]))
    assert totals(hists["met"]) == pytest.approx(6.0)


def test_per_object_fill_with_selections():
    events = make_events()
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 10, "lo": 0, "hi": 400,
                       "per_object": True, "selections": ["good", "highmet"]}}
    sels = {"good": {"expression": "events.Jet.pt > 60", "level": "object"},
            "highmet": {"expression": "events.met > 50"}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, sels, events, "s", np.array([1.0, 1.0, 1.0]))
    # events 2 and 3 pass met cut; jets > 60: none in event 2, the 200 in event 3
    assert totals(hists["jet_pt"]) == pytest.approx(1.0)


def test_variable_binning_edges():
    defs = {"met": {"expression": "events.met", "edges": [0, 50, 100, 400]}}
    h = build_histograms(defs, ["s"])["met"]
    axis = h.axes["x"]
    assert list(axis.edges) == [0, 50, 100, 400]
    fill_histograms({"met": h}, defs, {}, make_events(), "s", np.ones(3))
    assert list(h[{"dataset": "s"}].view().value) == [1.0, 1.0, 1.0]


def test_jagged_weight_per_object_fill():
    events = make_events()
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400,
                       "per_object": True, "weight": "events.Jet.pt / 100.0"}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events, "s", np.ones(3))
    # each jet weighted by pt/100: 0.5 + 1.0 + 2.0
    assert totals(hists["jet_pt"]) == pytest.approx(3.5)


def test_jagged_weight_event_level_takes_product():
    events = make_events()
    defs = {"met": {"expression": "events.met", "bins": 10, "lo": 0, "hi": 200,
                    "weight": "events.Jet.pt / 100.0"}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events, "s", np.ones(3))
    # products per event: 0.5*1.0, empty -> 1, 2.0
    assert totals(hists["met"]) == pytest.approx(0.5 + 1.0 + 2.0)


def test_at_map_reference_fill_only():
    events = make_events()
    rw = Reweighter("m", {"variable": "events.Jet.pt", "edges": [0, 100, 400],
                          "weights": [0.0, 1.0], "level": "object",
                          "fill_only": True})
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400,
                       "per_object": True, "weight": "@m"}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events, "s", np.ones(3), reweighters={"m": rw})
    # jets with pt < 100 zeroed -> 100 and 200 remain
    assert totals(hists["jet_pt"]) == pytest.approx(2.0)


def test_mismatched_object_weight_raises():
    jets = ak.Array([[{"pt": 50.0}], [{"pt": 60.0}, {"pt": 70.0}]])
    other = ak.Array([[1.0, 2.0], [3.0]])  # different structure
    events = ak.zip({"Jet": jets, "w": other}, depth_limit=1)
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400,
                       "per_object": True, "weight": "events.w"}}
    hists = build_histograms(defs, ["s"])
    with pytest.raises(ValueError, match="object structure"):
        fill_histograms(hists, defs, {}, events, "s", np.ones(2))


def test_none_values_dropped_weights_stay_aligned():
    events = make_events()
    defs = {"lead": {"expression": "ak.firsts(events.Jet.pt)",
                     "bins": 4, "lo": 0, "hi": 400}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events, "s", np.array([1.0, 5.0, 2.0]))
    # event 2 has no jets -> None -> dropped; weights 1 and 2 remain
    assert totals(hists["lead"]) == pytest.approx(3.0)
