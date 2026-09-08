"""Fill logic tests on synthetic (ROOT-free) awkward events."""

import awkward as ak
import numpy as np
import pytest

from suep_plot.histograms import (_compile_expr, attach_selection_columns,
                                  build_histograms, fill_histograms)
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


# ── Selections as columns ──────────────────────────────────────────
# Every selections.yaml mask is attached to events under its own name, so a
# histogram can count what a cut keeps without restating the cut.

def test_a_selection_becomes_a_column_of_its_own_name():
    events = make_events()
    sels = {"good_jets": {"level": "object",
                          "expression": "events.Jet.pt > 60"},
            "high_met": {"expression": "events.met > 50"}}

    out = attach_selection_columns(events, sels)

    assert ak.to_list(out.good_jets) == [[False, True], [], [True]]
    assert ak.to_list(out.high_met) == [False, True, True]


def test_the_column_is_what_a_multiplicity_histogram_fills_from():
    """The documented idiom: ak.sum(events.<selection>, axis=1).

    This is the whole point of the feature -- the count comes from the cut
    itself, so the plot cannot disagree with the selection it is named after.
    """
    events = make_events()
    sels = {"good_jets": {"level": "object", "expression": "events.Jet.pt > 60"}}
    events = attach_selection_columns(events, sels)

    defs = {"n_good_jets": {"expression": "ak.sum(events.good_jets, axis=1)",
                            "bins": 5, "lo": -0.5, "hi": 4.5}}
    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, sels, events, "s")

    # 2-jet event has 1 passing, empty event 0, 1-jet event 1.
    counts = hists["n_good_jets"][{"dataset": "s"}].values()
    assert list(counts) == [1.0, 2.0, 0.0, 0.0, 0.0]


def test_an_object_selection_column_stays_jagged():
    """ak.sum(axis=1) needs the per-object structure, not an event-level any()."""
    events = attach_selection_columns(
        make_events(),
        {"good_jets": {"level": "object", "expression": "events.Jet.pt > 60"}})

    assert events.good_jets.ndim == 2
    assert ak.to_list(ak.sum(events.good_jets, axis=1)) == [1, 0, 1]


def test_a_selection_that_does_not_evaluate_is_skipped_not_fatal():
    """Config sets are shared across samples with different content.

    A broken expression is reported by the one-time validation pass; it must not
    take down every chunk of every sample here.
    """
    events = attach_selection_columns(
        make_events(),
        {"nope": {"expression": "events.NoSuchCollection.pt > 0"},
         "fine": {"expression": "events.met > 50"}})

    assert "nope" not in events.fields
    assert "fine" in events.fields


def test_a_selection_never_shadows_an_existing_field(capsys):
    """A selection named after a real branch must not redefine it.

    Overwriting events.Jet would silently change what every other expression
    in the config set means.
    """
    events = attach_selection_columns(
        make_events(), {"Jet": {"expression": "events.met > 50"}})

    assert events.Jet.fields == ["pt", "eta"]        # the collection, untouched
    assert "already a field" in capsys.readouterr().out


def test_none_entries_become_false():
    """A mask with missing values must be usable as a mask.

    ak.firsts on an empty event gives None; left as None it would poison both
    ak.sum and the object filtering.
    """
    events = attach_selection_columns(
        make_events(), {"lead_jet_hard": {"expression": "ak.firsts(events.Jet.pt) > 60"}})

    assert ak.to_list(events.lead_jet_hard) == [False, False, True]


# ── Nested scopes in config expressions ────────────────────────────
# `events`/`ak`/`np` are passed as eval's *globals*, not locals: a name in
# eval's locals is invisible to a nested scope, so a lambda or comprehension in
# a config expression used to fail with "NameError: name 'ak' is not defined".

def test_a_lambda_in_an_expression_can_see_the_namespace():
    """The idiom for reusing an intermediate without repeating its expression."""
    events = make_events()
    expr = ("(lambda p: ak.sum(p.a + p.b, axis=1))("
            "ak.combinations(events.Jet.pt, 2, axis=1, fields=['a', 'b']))")

    # The 2-jet event has one pair (50 + 100); the others have none.
    assert ak.to_list(_compile_expr(expr)(events)) == [150.0, 0.0, 0.0]


def test_a_comprehension_in_an_expression_can_see_the_namespace():
    events = make_events()
    expr = "ak.concatenate([ak.num(events.Jet), ak.num(events.Jet)], axis=0)"
    assert ak.to_list(_compile_expr(expr)(events)) == [2, 0, 1, 2, 0, 1]

    # A genuine comprehension, which is its own scope.
    expr = "np.array([len(x) for x in ak.to_list(events.Jet.pt)])"
    assert list(_compile_expr(expr)(events)) == [2, 0, 1]


def test_the_builtins_are_still_restricted():
    """Moving the names to globals must not widen what an expression can reach."""
    events = make_events()
    for expr in ("__import__('os').listdir('.')", "open('/etc/passwd')",
                 "eval('1+1')"):
        with pytest.raises(NameError):
            _compile_expr(expr)(events)

    # The four that are provided still work.
    assert _compile_expr("max(1, 2)")(events) == 2
    assert _compile_expr("abs(-3)")(events) == 3


def test_the_max_dphi_idiom():
    """The pairwise-dphi expression used by configs_mds_objects.

    Pins two things: the wrap into (-pi, pi], and that an event with fewer than
    two objects yields None (dropped at fill) rather than a spurious 0.
    """
    clusters = ak.Array([
        [{"phi": 0.1}, {"phi": 3.0}, {"phi": -3.0}],   # widest pair: 0.1 vs -3.0
        [{"phi": 0.5}],                                 # no pair
        [],                                             # no pair
    ])
    events = ak.zip({"cscCluster": clusters}, depth_limit=1)
    expr = ("ak.max(abs((np.subtract(*ak.unzip(ak.combinations("
            "events.cscCluster.phi, 2, axis=1))) + np.pi) % (2 * np.pi) - np.pi), "
            "axis=1)")

    got = ak.to_list(_compile_expr(expr)(events))
    assert got[0] == pytest.approx(3.1)      # |0.1 - (-3.0)| = 3.1 < pi
    assert got[1] is None and got[2] is None

    # Never exceeds pi, whatever the inputs.
    wide = ak.zip({"cscCluster": ak.Array([[{"phi": -3.14}, {"phi": 3.14}]])},
                  depth_limit=1)
    assert _compile_expr(expr)(wide)[0] == pytest.approx(0.0034, abs=1e-3)
