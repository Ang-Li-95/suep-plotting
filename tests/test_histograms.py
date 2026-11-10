"""Fill logic tests on synthetic (ROOT-free) awkward events."""

import awkward as ak
import numpy as np
import pytest

from suep_plot.histograms import (FillError, _compile_expr, attach_selection_columns,
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


def test_a_selection_that_does_not_evaluate_is_fatal():
    """A mask silently missing on one sample is a cut silently not applied."""
    with pytest.raises(FillError, match="selection 'nope'.*sample 'sig'"):
        attach_selection_columns(
            make_events(),
            {"nope": {"expression": "events.NoSuchCollection.pt > 0"},
             "fine": {"expression": "events.met > 50"}}, "sig")


def test_selections_may_read_each_other_in_any_order():
    events = attach_selection_columns(make_events(), {
        "both": {"expression": "events.hard & events.highmet"},
        "hard": {"expression": "ak.any(events.Jet.pt > 60, axis=1)"},
        "highmet": {"expression": "events.met > 50"},
    })

    assert ak.to_list(events.both) == [False, False, True]


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


# ── per object or per event: read off the expression ──────────────


def test_per_object_is_read_off_the_expression():
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400},
            "njet": {"expression": "ak.num(events.Jet)", "bins": 3, "lo": 0, "hi": 3}}
    hists = build_histograms(defs, ["s"])
    layout = {}

    fill_histograms(hists, defs, {}, make_events(), "s", np.ones(3), layout=layout)

    assert layout == {"jet_pt": True, "njet": False}
    assert totals(hists["jet_pt"]) == 3          # one entry per jet
    assert totals(hists["njet"]) == 3            # one entry per event


def test_an_explicit_per_object_that_contradicts_the_expression_is_fatal():
    defs = {"met": {"expression": "events.met", "bins": 2, "lo": 0, "hi": 200,
                    "per_object": True}}

    with pytest.raises(FillError, match="per_object: True"):
        fill_histograms(build_histograms(defs, ["s"]), defs, {}, make_events(), "s",
                        np.ones(3))


def test_an_object_selection_needs_no_level_key():
    defs = {"jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400,
                       "selections": ["good"]}}
    sels = {"good": {"expression": "events.Jet.pt > 60"}}
    hists = build_histograms(defs, ["s"])

    fill_histograms(hists, defs, sels, make_events(), "s", np.ones(3))

    assert totals(hists["jet_pt"]) == 2      # the 100 and the 200 GeV jets


def test_an_object_selection_on_a_per_event_histogram_is_fatal():
    defs = {"met": {"expression": "events.met", "bins": 4, "lo": 0, "hi": 400,
                    "selections": ["good"]}}
    sels = {"good": {"expression": "events.Jet.pt > 60"}}

    with pytest.raises(FillError, match=r"histogram 'met' fills once per event.*"
                                        r"\['good'\].*ak\.any\(events\.good, axis=1\)"):
        fill_histograms(build_histograms(defs, ["s"]), defs, sels, make_events(),
                        "s", np.ones(3))


def test_the_event_level_cut_is_a_selection_of_its_own():
    defs = {"met": {"expression": "events.met", "bins": 4, "lo": 0, "hi": 400,
                    "selections": ["has_good"]}}
    sels = {"good": {"expression": "events.Jet.pt > 60"},
            "has_good": {"expression": "ak.any(events.good, axis=1)"}}
    events = attach_selection_columns(make_events(), sels)
    hists = build_histograms(defs, ["s"])

    fill_histograms(hists, defs, sels, events, "s", np.ones(3))

    assert totals(hists["met"]) == 2         # events with >= 1 passing jet


def test_a_histogram_that_does_not_evaluate_is_fatal_and_named():
    defs = {"bad": {"expression": "events.NoSuch.pt", "bins": 2, "lo": 0, "hi": 1}}

    with pytest.raises(FillError, match="histogram 'bad'.*sample 's'"):
        fill_histograms(build_histograms(defs, ["s"]), defs, {}, make_events(), "s",
                        np.ones(3))


def test_a_2d_histogram_mixing_per_object_and_per_event_axes_is_fatal():
    defs = {"mix": {"expression_x": "events.Jet.pt", "expression_y": "events.met",
                    "bins_x": 2, "lo_x": 0, "hi_x": 400,
                    "bins_y": 2, "lo_y": 0, "hi_y": 200}}

    with pytest.raises(FillError, match="both"):
        fill_histograms(build_histograms(defs, ["s"]), defs, {}, make_events(), "s",
                        np.ones(3))


# ── config files ──────────────────────────────────────────────────


def _yaml(path, obj):
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))
    return path


def test_a_selection_can_be_just_its_expression(tmp_path):
    from suep_plot.histograms import load_selection_defs

    p = _yaml(tmp_path / "s.yaml", {"highmet": "events.met > 50",
                                    "good": {"expression": "events.Jet.pt > 60"}})

    assert load_selection_defs(p) == {"highmet": {"expression": "events.met > 50"},
                                      "good": {"expression": "events.Jet.pt > 60"}}


# ── the variable catalog ──────────────────────────────────────────

CATALOG = {
    "met": {"expression": "events.met", "bins": 10, "lo": 0, "hi": 200,
            "label": "MET", "log_y": True},
    "jet_pt": {"expression": "events.Jet.pt", "bins": 4, "lo": 0, "hi": 400,
               "label": "jet pT"},
    "jet_eta": {"expression": "events.Jet.eta", "edges": [-3, 0, 3],
                "label": "jet eta"},
}


def _region(tmp_path, *entries, variables=None):
    """The histograms of one region whose per-event list is *entries*."""
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"regions": {"r": {"evt_vars": list(entries)}}})
    return load_histogram_defs(p, variables=CATALOG if variables is None else variables)


def test_an_entry_takes_its_variable_from_the_catalog(tmp_path):
    defs = _region(tmp_path, "met", {"var": "met", "name": "met_wide",
                                     "bins": [20, 0, 400], "log_y": False})

    assert defs["r/met"] == {"expression": "events.met", "bins": 10, "lo": 0,
                             "hi": 200, "label": "MET", "log_y": True,
                             "selections": [], "annotation": "r"}
    # What the entry writes wins, key by key.
    wide = defs["r/met_wide"]
    assert (wide["bins"], wide["hi"], wide["log_y"]) == (20, 400, False)


def test_an_entry_can_move_one_end_of_the_range(tmp_path):
    met = _region(tmp_path, {"var": "met", "hi": 400})["r/met"]

    assert (met["bins"], met["lo"], met["hi"]) == (10, 0, 400)


def test_edges_and_bins_replace_each_other(tmp_path):
    defs = _region(tmp_path,
                   {"var": "met", "name": "met_edges", "edges": [0, 50, 200]},
                   {"var": "jet_eta", "name": "eta_bins", "bins": [6, -3, 3]})

    assert defs["r/met_edges"]["edges"] == [0, 50, 200]
    assert not {"bins", "lo", "hi"} & set(defs["r/met_edges"])
    assert defs["r/eta_bins"]["bins"] == 6 and "edges" not in defs["r/eta_bins"]


def test_a_2d_entry_retunes_one_axis(tmp_path):
    defs = _region(tmp_path, {"name": "rz", "x": "met",
                              "y": {"var": "jet_pt", "bins": [8, 0, 400]},
                              "log_z": True})

    rz = defs["r/rz"]
    assert (rz["expression_x"], rz["bins_x"]) == ("events.met", 10)
    assert (rz["expression_y"], rz["bins_y"], rz["log_z"]) == ("events.Jet.pt", 8, True)
    assert "log_y" not in rz             # 1D display defaults stay on 1D


def test_an_unknown_variable_names_the_closest(tmp_path):
    from suep_plot.config import ConfigError

    with pytest.raises(ConfigError, match="did you mean jet_pt"):
        _region(tmp_path, "jet_ptt")


def test_an_unknown_axis_key_is_reported(tmp_path):
    from suep_plot.config import ConfigError

    with pytest.raises(ConfigError, match="unknown axis key"):
        _region(tmp_path, {"name": "rz", "x": {"var": "met", "binz": 3}, "y": "jet_pt"})


def test_a_variable_without_binning_is_reported(tmp_path):
    from suep_plot.config import ConfigError

    with pytest.raises(ConfigError, match="no binning"):
        _region(tmp_path, "met", variables={"met": {"expression": "events.met"}})


@pytest.mark.parametrize("entry", [
    {"name": "h", "expression": "events.met", "bins": [10, 0, 200]},
    {"name": "h", "x": {"expression": "events.met", "bins": [10, 0, 200]}, "y": "met"},
    {"var": "met", "variants": {"m": {"selections": ["good"]}}},
])
def test_an_entry_cannot_define_an_expression_or_variants(tmp_path, entry):
    from suep_plot.config import ConfigError

    with pytest.raises(ConfigError, match="variables.yaml"):
        _region(tmp_path, entry)


def test_per_object_is_not_a_key_of_an_entry(tmp_path):
    from suep_plot.config import ConfigError

    with pytest.raises(ConfigError, match="'per_object' is not a key"):
        _region(tmp_path, {"var": "jet_pt", "per_object": True})


def test_a_histogram_outside_regions_is_refused(tmp_path):
    from suep_plot.config import ConfigError
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"met": {"expression": "events.met",
                                            "bins": [10, 0, 200]}})

    with pytest.raises(ConfigError, match="inside 'regions' only.*met"):
        load_histogram_defs(p, variables=CATALOG)


def test_variant_sets_are_no_longer_a_directive(tmp_path):
    from suep_plot.config import ConfigError
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"_variant_sets": {"m": {}}})

    with pytest.raises(ConfigError, match="unknown directive"):
        load_histogram_defs(p, variables=CATALOG)


def test_a_selection_given_as_a_string_is_one_selection(tmp_path):
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"regions": {"r": {
        "evt_sel": "highmet",
        "objects": {"hard": {"cut": "good", "vars": [
            {"var": "jet_pt", "selections": "central"}]}}}}})

    assert load_histogram_defs(p, variables=CATALOG)["r/hard/jet_pt"]["selections"] \
        == ["highmet", "good", "central"]


def test_a_set_retunes_one_key_of_a_global_variable(tmp_path, monkeypatch):
    from suep_plot import histograms

    glob = _yaml(tmp_path / "variables.yaml",
                 {"met": {"expression": "events.met", "bins": [10, 0, 200],
                          "label": "MET"}})
    monkeypatch.setattr(histograms, "GLOBAL_VARIABLES", glob)
    _yaml(tmp_path / "set" / "variables.yaml", {"met": {"bins": [20, 0, 200]}})

    met = histograms.load_variables(tmp_path / "set")["met"]
    assert (met["bins"], met["hi"], met["label"]) == (20, 200, "MET")


def test_a_variable_with_an_unknown_key_is_reported(tmp_path, monkeypatch):
    from suep_plot import histograms
    from suep_plot.config import ConfigError

    glob = _yaml(tmp_path / "variables.yaml",
                 {"met": {"expression": "events.met", "bins": [10, 0, 200],
                          "titel": "MET"}})
    monkeypatch.setattr(histograms, "GLOBAL_VARIABLES", glob)

    with pytest.raises(ConfigError, match="unknown key"):
        histograms.load_variables()


# ── regions ───────────────────────────────────────────────────────


def test_regions_expand_to_one_histogram_per_cut_and_variable(tmp_path):
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"regions": {
        "jets": {
            "label": "Jets",
            "evt_sel": ["highmet"],
            "evt_vars": ["met"],
            "objects": {
                "all": {"vars": ["jet_pt", ["jet_pt", "jet_eta"]]},
                "central": {"cut": ["central"], "label": "|eta| < 1",
                            "vars": ["jet_pt", {"var": "jet_pt", "name": "jet_pt_fine",
                                                "bins": [40, 0, 400]}]},
            }}}})
    defs = load_histogram_defs(p, variables=CATALOG)

    assert sorted(defs) == ["jets/all/jet_pt", "jets/all/jet_pt_vs_jet_eta",
                            "jets/central/jet_pt", "jets/central/jet_pt_fine",
                            "jets/met"]
    assert defs["jets/central/jet_pt"]["selections"] == ["highmet", "central"]
    assert defs["jets/met"]["selections"] == ["highmet"]
    assert defs["jets/central/jet_pt"]["annotation"] == "Jets · |eta| < 1"
    assert defs["jets/all/jet_pt"]["annotation"] == "Jets"
    assert defs["jets/central/jet_pt_fine"]["bins"] == 40
    assert defs["jets/all/jet_pt_vs_jet_eta"]["edges_y"] == [-3, 0, 3]


def test_a_region_family_is_written_once_with_repeat(tmp_path):
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {
        "_axes": {"systems": [{"SYS": "a"}, {"SYS": "b"}]},
        "regions": {"_repeat": [{"over": "systems", "defs": {
            "<SYS>": {"objects": {"all": {"vars": ["jet_pt"]}}}}}]},
    })

    assert sorted(load_histogram_defs(p, variables=CATALOG)) == \
        ["a/all/jet_pt", "b/all/jet_pt"]


def test_a_child_set_adds_a_cut_to_an_inherited_region(tmp_path):
    from suep_plot.histograms import load_histogram_defs

    _yaml(tmp_path / "base" / "histograms.yaml", {"regions": {
        "jets": {"objects": {"all": {"vars": ["jet_pt"]}}}}})
    child = _yaml(tmp_path / "child" / "histograms.yaml", {
        "_extends": "../base",
        "regions": {"jets": {"objects": {"central": {"cut": "central",
                                                     "vars": ["jet_pt"]}}}}})

    assert sorted(load_histogram_defs(child, variables=CATALOG)) == \
        ["jets/all/jet_pt", "jets/central/jet_pt"]


def test_a_variable_listed_twice_under_one_cut_is_reported(tmp_path):
    from suep_plot.config import ConfigError
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"regions": {
        "jets": {"objects": {"all": {"vars": ["jet_pt", "jet_pt"]}}}}})

    with pytest.raises(ConfigError, match="listed twice"):
        load_histogram_defs(p, variables=CATALOG)


def test_a_misspelled_region_key_is_reported(tmp_path):
    from suep_plot.config import ConfigError
    from suep_plot.histograms import load_histogram_defs

    p = _yaml(tmp_path / "h.yaml", {"regions": {"jets": {"event": ["x"]}}})

    with pytest.raises(ConfigError, match="unknown key"):
        load_histogram_defs(p, variables=CATALOG)


def test_region_histograms_fill_after_both_cuts():
    from suep_plot.histograms import expand_regions

    defs = expand_regions({"jets": {"evt_sel": ["highmet"], "objects": {
        "hard": {"cut": ["good"], "vars": ["jet_pt"]}}}}, CATALOG)
    sels = {"good": {"expression": "events.Jet.pt > 60"},
            "highmet": {"expression": "events.met > 50"}}
    hists = build_histograms(defs, ["s"])

    fill_histograms(hists, defs, sels, make_events(), "s", np.ones(3))

    # met > 50: events 2 and 3; jets > 60 there: only the 200 GeV one
    assert totals(hists["jets/hard/jet_pt"]) == 1
