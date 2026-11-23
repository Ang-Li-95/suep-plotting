"""Corrections: the event weight, and a correction named as a histogram's weight."""

import awkward as ak
import numpy as np
import pytest

from suep_plot.checks import reference_errors
from suep_plot.corrections import apply_corrections, correction_weights
from suep_plot.fill import build_histograms, fill_histograms
from suep_plot.histdefs import as_histograms


def muon_sf(pt):
    """A stand-in for a correctionlib evaluator: 0.9 above 20 GeV, 0.5 below."""
    return np.where(np.asarray(pt) > 20, 0.9, 0.5)


def correctors(**keys):
    """What build_correctors returns, for one per-muon scale factor."""
    cfg = {"inputs": ["events.Muon.pt"], "kind": "object_sf", "apply_to": "all", **keys}
    return {"muon_sf": (muon_sf, cfg)}


def events():
    return ak.Array({"met": [10.0, 20.0, 30.0],
                     "Muon": ak.zip({"pt": [[30.0, 10.0], [], [40.0]]})})


class Collected:
    """The part of coffea's Weights that apply_corrections uses."""

    def __init__(self):
        self.added = {}

    def add(self, name, values):
        self.added[name] = values


MUON_PT = {"muon_pt": {"expression": "events.Muon.pt", "bins": 5, "lo": 0, "hi": 50,
                       "weight": "@muon_sf"}}
MET = {"met": {"expression": "events.met", "bins": 3, "lo": 5, "hi": 35,
               "weight": "@muon_sf"}}


def filled(defs, group="signal", is_data=False, sample="s", **keys):
    defs = as_histograms(defs)
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, {}, events(), "s", named_weights=correction_weights(
        correctors(**keys), sample, group, is_data))
    (h,) = hists.values()
    return h.values().tolist()[0]


def test_a_per_object_scale_factor_enters_the_event_weight_as_a_product():
    weights = Collected()

    apply_corrections(correctors(), events(), "s", "signal", False, weights)

    # 0.9 * 0.5, no muon -> 1, 0.9
    assert weights.added["muon_sf"].tolist() == pytest.approx([0.45, 1.0, 0.9])


def test_a_fill_only_correction_stays_out_of_the_event_weight():
    weights = Collected()

    apply_corrections(correctors(fill_only=True), events(), "s", "signal", False,
                      weights)

    assert weights.added == {}


def test_a_per_object_histogram_weights_each_object_with_its_own_factor():
    # bins of 10 GeV: the 10 GeV muon (0.5), the 30 GeV one (0.9), the 40 GeV one (0.9)
    assert filled(MUON_PT) == pytest.approx([0.0, 0.5, 0.0, 0.9, 0.9])


def test_a_per_event_histogram_gets_the_product_over_the_objects():
    assert filled(MET) == pytest.approx([0.45, 1.0, 0.9])


@pytest.mark.parametrize("group, is_data, keys", [
    ("data", True, {}),                                  # data is never corrected
    ("background", False, {"apply_to": ["signal"]}),     # not one of its groups
])
def test_a_sample_the_correction_does_not_concern_fills_with_weight_one(
        group, is_data, keys):
    assert filled(MUON_PT, group, is_data, **keys) == [0.0, 1.0, 0.0, 1.0, 1.0]


def test_a_correction_reaches_data_only_when_samples_names_it():
    weights = Collected()
    apply_corrections(correctors(), events(), "data_2024", "", True, weights)
    assert weights.added == {}                           # simulation only

    named = correctors(samples=["data_2024"])
    apply_corrections(named, events(), "data_2024", "", True, weights)
    assert weights.added["muon_sf"].tolist() == pytest.approx([0.45, 1.0, 0.9])
    # ... and then it is for those samples alone, simulation included.
    weights = Collected()
    apply_corrections(named, events(), "dy", "background", False, weights)
    assert weights.added == {}

    assert filled(MUON_PT, "", True, sample="data_2024", samples="data_2024") \
        == pytest.approx([0.0, 0.5, 0.0, 0.9, 0.9])


def test_a_weight_over_another_collection_is_fatal():
    from suep_plot.fill import FillError

    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 2, "lo": 0, "hi": 100,
                       "weight": "@muon_sf"}}
    ev = ak.with_field(events(), ak.zip({"pt": [[50.0], [60.0], []]}), "Jet")
    typed = as_histograms(defs)

    with pytest.raises(FillError, match="does not match the object structure"):
        fill_histograms(build_histograms(typed, ["s"]), typed, {}, ev, "s",
                        named_weights=correction_weights(correctors(), "s",
                                                         "signal", False))


def test_a_histogram_weight_may_name_a_correction():
    hists = as_histograms(MUON_PT)

    assert reference_errors(hists, {}, {}, {"muon_sf": {}}) == []
    (error,) = reference_errors(hists, {}, {}, {})
    assert "unknown weight '@muon_sf'" in error


def test_a_name_cannot_be_both_a_reweight_and_a_correction():
    (error,) = reference_errors({}, {}, {"muon_sf": object()}, {"muon_sf": {}})

    assert "both a reweight" in error and "'@muon_sf' would be ambiguous" in error
