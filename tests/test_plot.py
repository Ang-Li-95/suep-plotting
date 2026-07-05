"""Plot-side helpers: flow folding, xs scaling, merging, cutflow tables."""

import pickle

import hist
import numpy as np
import pytest

from suep_plot.plot import (
    _fold_flow,
    apply_xs_scaling,
    merge_results,
    write_cutflow,
)


def make_hist(sample, vals, weights=None):
    h = hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.Regular(2, 0, 100, name="x"),
        storage=hist.storage.Weight(),
    )
    h.fill(dataset=sample, x=vals, weight=weights)
    return h


def test_fold_flow():
    h = make_hist("s", [-10, 25, 75, 150, 200])
    sh = _fold_flow(h[{"dataset": "s"}])
    assert list(sh.view().value) == [2.0, 3.0]         # under->first, over->last
    assert sh.sum(flow=True).value == sh.sum().value   # flow bins emptied


def test_apply_xs_scaling():
    h = make_hist("mc", [25, 75])
    hists = {"h": h}
    sample_defs = {"mc": {"xs": 2.0}, "data": {"is_data": True}}
    apply_xs_scaling(hists, sample_defs, {"mc": 100.0}, lumi=5.0)
    # factor = 2 * 5 * 1000 / 100 = 100
    assert list(h[{"dataset": "mc"}].view().value) == [100.0, 100.0]


def test_merge_results(tmp_path):
    for s, vals in (("a", [25]), ("b", [75, 75])):
        payload = {"histograms": {"met": make_hist(s, vals)},
                   "samples": {s: {"label": s}},
                   "hist_defs": {"met": {"bins": 2}},
                   "sumw": {s: 1.0}, "nevents": {s: len(vals)},
                   "cutflow": {s: {"total": {"raw": len(vals), "wtd": float(len(vals))}}}}
        with open(tmp_path / f"{s}.pkl", "wb") as f:
            pickle.dump(payload, f)

    merged = merge_results([str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")])
    h = merged["histograms"]["met"]
    assert set(h.axes["dataset"]) == {"a", "b"}
    assert h.sum(flow=True).value == 3.0
    assert set(merged["cutflow"]) == {"a", "b"}


def test_write_cutflow(tmp_path):
    cutflow = {"sig": {"total": {"raw": 100, "wtd": 99.5},
                       "baseline": {"raw": 40, "wtd": 39.0}}}
    write_cutflow(cutflow, str(tmp_path))
    txt = (tmp_path / "cutflow.txt").read_text()
    assert "baseline" in txt and "sig" in txt
    csv = (tmp_path / "cutflow.csv").read_text().strip().splitlines()
    assert csv[0] == "sample,selection,raw,weighted"
    assert csv[2].startswith("sig,baseline,40,")
