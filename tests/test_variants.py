"""Variant expansion: one definition -> the (variable x selection) matrix."""

import pytest
import yaml

from suep_plot.histograms import expand_variants


def _expand(text):
    return expand_variants(yaml.safe_load(text) or {})


BASE = """
csc_cluster_size:
  expression: "events.cscCluster.size"
  per_object: true
  bins: 50
  lo: 0
  hi: 500
  label: 'CSC cluster N'
"""


def test_definitions_without_variants_pass_through_unchanged():
    assert _expand(BASE) == yaml.safe_load(BASE)


def test_inline_variants_append_selections_and_decorate_labels():
    out = _expand(BASE + """
  selections: [has_csc_cluster]
  variants:
    matched:   {selections: [csc_cluster_matched],   label_prefix: "matched "}
    unmatched: {selections: [csc_cluster_unmatched], label_suffix: " (fakes)"}
""")

    assert list(out) == ["csc_cluster_size", "csc_cluster_size_matched",
                         "csc_cluster_size_unmatched"]
    assert out["csc_cluster_size"]["selections"] == ["has_csc_cluster"]
    assert out["csc_cluster_size_matched"]["selections"] == \
        ["has_csc_cluster", "csc_cluster_matched"]
    assert out["csc_cluster_size_matched"]["label"] == "matched CSC cluster N"
    assert out["csc_cluster_size_unmatched"]["label"] == "CSC cluster N (fakes)"
    # fill-time keys are inherited verbatim
    assert out["csc_cluster_size_matched"]["bins"] == 50
    assert out["csc_cluster_size_matched"]["expression"] == "events.cscCluster.size"


def test_named_variant_set_is_reusable_and_underscore_keys_are_not_histograms():
    out = _expand("""
_variant_sets:
  csc_match:
    matched: {selections: [csc_cluster_matched]}

csc_cluster_size:
  expression: "events.cscCluster.size"
  bins: 5
  lo: 0
  hi: 5
  variants: csc_match

csc_cluster_eta:
  expression: "events.cscCluster.eta"
  bins: 5
  lo: -3
  hi: 3
  variants: csc_match
""")

    assert list(out) == ["csc_cluster_size", "csc_cluster_size_matched",
                         "csc_cluster_eta", "csc_cluster_eta_matched"]
    assert "_variant_sets" not in out


def test_keep_base_false_drops_the_unselected_histogram():
    out = _expand(BASE + """
  keep_base: false
  variants:
    matched: {selections: [csc_cluster_matched]}
""")

    assert list(out) == ["csc_cluster_size_matched"]
    assert "keep_base" not in out["csc_cluster_size_matched"]


def test_variant_overrides_any_other_key():
    out = _expand(BASE + """
  log_y: true
  variants:
    matched: {log_y: false, hi: 100}
""")

    assert out["csc_cluster_size_matched"]["log_y"] is False
    assert out["csc_cluster_size_matched"]["hi"] == 100
    assert out["csc_cluster_size"]["hi"] == 500


def test_unknown_variant_set_is_fatal():
    with pytest.raises(SystemExit, match="unknown variant set 'nope'"):
        _expand(BASE + "  variants: nope\n")


def test_variant_colliding_with_an_explicit_histogram_is_fatal():
    with pytest.raises(SystemExit, match="csc_cluster_size_matched"):
        _expand(BASE + """
  variants:
    matched: {selections: [csc_cluster_matched]}

csc_cluster_size_matched:
  expression: "events.cscCluster.size"
  bins: 5
  lo: 0
  hi: 5
""")


def test_empty_variants_mapping_is_fatal():
    with pytest.raises(SystemExit, match="non-empty mapping"):
        _expand(BASE + "  variants: {}\n")


def test_expanded_variants_fill_with_their_selections_applied():
    """End-to-end: expansion feeds the normal build/fill path."""
    import awkward as ak
    import numpy as np

    from suep_plot.histograms import build_histograms, fill_histograms

    clusters = ak.Array([
        [{"size": 100.0, "matched": True}, {"size": 20.0, "matched": False}],
        [{"size": 300.0, "matched": False}],
    ])
    events = ak.zip({"cscCluster": clusters}, depth_limit=1)

    defs = _expand("""
csc_cluster_size:
  expression: "events.cscCluster.size"
  per_object: true
  bins: 40
  lo: 0
  hi: 400
  variants:
    matched:   {selections: [csc_matched]}
    unmatched: {selections: [csc_unmatched]}
""")
    sels = {
        "csc_matched": {"expression": "events.cscCluster.matched", "level": "object"},
        "csc_unmatched": {"expression": "~events.cscCluster.matched", "level": "object"},
    }

    hists = build_histograms(defs, ["s"])
    fill_histograms(hists, defs, sels, events, "s", np.array([1.0, 1.0]))

    def total(name):
        return hists[name][{"dataset": "s"}].sum(flow=True).value

    assert total("csc_cluster_size") == pytest.approx(3.0)            # all clusters
    assert total("csc_cluster_size_matched") == pytest.approx(1.0)    # the one matched
    assert total("csc_cluster_size_unmatched") == pytest.approx(2.0)  # the other two


def test_isolation_cuts_are_identical_in_the_grid_and_data_configs():
    """The two studies are meant to be overlaid, so the cuts must not drift."""
    from pathlib import Path

    from suep_plot.histograms import load_selection_defs

    repo = Path(__file__).resolve().parent.parent
    sels = {}
    for cfg in ("configs_mds_signal", "configs_mds_data"):
        # Through the loader, so a set that writes its cuts once under _repeat
        # is compared on the selections it actually produces.
        defs = load_selection_defs(repo / "configs" / cfg / "selections.yaml")
        sels[cfg] = {k: v for k, v in defs.items() if "_iso_" in k}

    grid, data = sels["configs_mds_signal"], sels["configs_mds_data"]
    assert set(grid) == set(data) and len(grid) == 6, sorted(set(grid) ^ set(data))
    for name in grid:
        assert grid[name] == data[name], f"{name} differs between the two configs"


def test_isolation_selections_keep_clusters_with_no_muon_or_jet():
    """drMuon/drJet carry the NO_OBJECT_DR sentinel when the event has none."""
    import awkward as ak
    import numpy as np
    from pathlib import Path

    from suep_plot.histograms import build_histograms, fill_histograms, load_histogram_defs

    repo = Path(__file__).resolve().parent.parent / "configs" / "configs_mds_data"
    sels = yaml.safe_load((repo / "selections.yaml").read_text())
    defs = load_histogram_defs(repo / "histograms.yaml")
    use = {k: defs[k] for k in ("csc_cluster_size", "csc_cluster_size_iso_muon")}

    clusters = ak.Array([
        [{"size": 100.0, "drMuon": 0.10, "drJet": 2.0}],   # close to a muon -> cut
        [{"size": 150.0, "drMuon": 1.50, "drJet": 2.0}],   # far -> kept
        [{"size": 200.0, "drMuon": 999.0, "drJet": 2.0}],  # no muon at all -> kept
    ])
    events = ak.zip({"cscCluster": clusters}, depth_limit=1)

    hists = build_histograms(use, ["s"])
    fill_histograms(hists, use, sels, events, "s", np.ones(3))

    def total(name):
        return hists[name][{"dataset": "s"}].sum(flow=True).value

    assert total("csc_cluster_size") == pytest.approx(3.0)
    assert total("csc_cluster_size_iso_muon") == pytest.approx(2.0)
