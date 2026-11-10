"""The isolation cuts of the shipped sets: shared, and safe without objects."""

import pytest


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
    # CSC and DT only: the default rpc_mode: match builds no RPC cluster.
    assert set(grid) == set(data) and len(grid) == 4, sorted(set(grid) ^ set(data))
    for name in grid:
        assert grid[name] == data[name], f"{name} differs between the two configs"


def test_isolation_selections_keep_clusters_with_no_muon_or_jet():
    """drMuon/drJet carry the NO_OBJECT_DR sentinel when the event has none."""
    import awkward as ak
    import numpy as np
    from pathlib import Path

    from suep_plot.histograms import (as_histograms, as_selections,
                                      build_histograms, fill_histograms,
                                      load_histogram_defs)

    from suep_plot.histograms import load_selection_defs

    repo = Path(__file__).resolve().parent.parent / "configs" / "configs_mds_data"
    # Through the loader: the study inherits these from ../common.
    sels = load_selection_defs(repo / "selections.yaml")
    defs = load_histogram_defs(repo / "histograms.yaml")
    use = {k: defs[k] for k in ("csc/all/csc_cluster_size",
                                "csc/iso_muon/csc_cluster_size")}

    clusters = ak.Array([
        [{"size": 100.0, "drMuon": 0.10, "drJet": 2.0}],   # close to a muon -> cut
        [{"size": 150.0, "drMuon": 1.50, "drJet": 2.0}],   # far -> kept
        [{"size": 200.0, "drMuon": 999.0, "drJet": 2.0}],  # no muon at all -> kept
    ])
    events = ak.zip({"cscCluster": clusters}, depth_limit=1)

    hists = build_histograms(as_histograms(use), ["s"])
    fill_histograms(hists, as_histograms(use), as_selections(sels), events, "s", np.ones(3))

    def total(name):
        return hists[name][{"dataset": "s"}].sum(flow=True).value

    assert total("csc/all/csc_cluster_size") == pytest.approx(3.0)
    assert total("csc/iso_muon/csc_cluster_size") == pytest.approx(2.0)
