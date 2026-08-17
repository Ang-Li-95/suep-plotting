"""Plot-side helpers: flow folding, xs scaling, merging, cutflow tables."""

import pickle

import hist
import numpy as np
import pytest

from suep_plot.plot import (
    _fold_flow,
    _prep_1d,
    _plot_efficiency,
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


def make_hist_2d(sample, xs, ys):
    h = hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.Regular(2, 0, 100, name="x"),
        hist.axis.Regular(2, 0, 100, name="y"),
        storage=hist.storage.Weight(),
    )
    h.fill(dataset=sample, x=xs, y=ys)
    return h


def test_fold_flow():
    h = make_hist("s", [-10, 25, 75, 150, 200])
    sh = _fold_flow(h[{"dataset": "s"}])
    assert list(sh.view().value) == [2.0, 3.0]         # under->first, over->last
    assert sh.sum(flow=True).value == sh.sum().value   # flow bins emptied


def test_prep_1d_folds_flow_by_default():
    """Every entry lands on the canvas unless the config opts out."""
    sh = make_hist("s", [-10, 25, 75, 150, 200])[{"dataset": "s"}]
    assert list(_prep_1d(sh, {}).view().value) == [2.0, 3.0]
    assert list(_prep_1d(sh, {"flow": "sum"}).view().value) == [2.0, 3.0]
    assert list(_prep_1d(sh, {"flow": "none"}).view().value) == [1.0, 1.0]


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


def test_merge_results_sums_parts(tmp_path):
    """File-split runs produce several pickles for the *same* sample
    (<sample>.part<k>.pkl); everything keyed by sample must sum, not
    overwrite."""
    for k, vals in ((0, [25]), (1, [75, 75])):
        payload = {"histograms": {"met": make_hist("s", vals)},
                   "samples": {"s": {"label": "s"}},
                   "hist_defs": {"met": {"bins": 2}},
                   "sumw": {"s": float(len(vals))},
                   "nevents": {"s": len(vals)},
                   "cutflow": {"s": {"total": {"raw": len(vals), "wtd": float(len(vals))},
                                     f"only_part{k}": {"raw": 1, "wtd": 1.0}}}}
        with open(tmp_path / f"s.part{k}.pkl", "wb") as f:
            pickle.dump(payload, f)

    merged = merge_results([str(tmp_path / "s.part0.pkl"), str(tmp_path / "s.part1.pkl")])
    assert merged["histograms"]["met"].sum(flow=True).value == 3.0
    assert merged["sumw"] == {"s": 3.0}
    assert merged["nevents"] == {"s": 3}
    assert merged["cutflow"]["s"]["total"] == {"raw": 3, "wtd": 3.0}
    # selections present in only one part are kept as-is
    assert merged["cutflow"]["s"]["only_part0"] == {"raw": 1, "wtd": 1.0}
    assert merged["cutflow"]["s"]["only_part1"] == {"raw": 1, "wtd": 1.0}


def test_plot_efficiency_2d(tmp_path):
    """A 2D num/den pair renders one efficiency map per sample."""
    h_den = make_hist_2d("sig", [25, 25, 75, 75], [25, 75, 25, 75])
    h_num = make_hist_2d("sig", [25, 75], [25, 75])
    cfg = {"type": "efficiency", "numerator": "num", "denominator": "den",
           "label_z": "eff"}
    n = _plot_efficiency("eff_map", cfg, {"num": h_num, "den": h_den},
                         {"sig": {"label": "sig"}}, str(tmp_path),
                         formats=("png",))
    assert n == 1
    assert (tmp_path / "eff_map.png").exists()


def test_write_cutflow(tmp_path):
    cutflow = {"sig": {"total": {"raw": 100, "wtd": 99.5},
                       "baseline": {"raw": 40, "wtd": 39.0}}}
    write_cutflow(cutflow, str(tmp_path))
    txt = (tmp_path / "cutflow.txt").read_text()
    assert "baseline" in txt and "sig" in txt
    csv = (tmp_path / "cutflow.csv").read_text().strip().splitlines()
    assert csv[0] == "sample,selection,raw,weighted"
    assert csv[2].startswith("sig,baseline,40,")


# ── suep-status: completion check and targeted resubmission ──────


def _fake_run(tmp_path, tasks, written):
    """A run directory as suep-submit leaves it, with *written* pickles present."""
    slurm = tmp_path / "slurm"
    (slurm / "filelists").mkdir(parents=True)
    lines = []
    for sample, part in tasks:
        fl = slurm / "filelists" / f"{sample}{'.part' + str(part) if part != '' else ''}.txt"
        fl.write_text("/some/file.root\n")
        lines.append(f"{sample}\t{fl}\t{part}")
    (slurm / "task_list.txt").write_text("\n".join(lines) + "\n")
    (slurm / "job.sh").write_text("#!/bin/bash\nconda activate mds\n")
    for name in written:
        with open(tmp_path / name, "wb") as f:
            pickle.dump({"histograms": {}}, f)
    return tmp_path


def test_status_reports_missing_tasks(tmp_path):
    from suep_plot.status import check

    out = _fake_run(tmp_path, [("sig", 0), ("sig", 1), ("bkg", "")],
                    written=["sig.part0.pkl", "bkg.pkl"])
    done, missing, corrupt = check(str(out))
    assert [t.pkl_name for t in done] == ["sig.part0.pkl", "bkg.pkl"]
    assert [t.index for t in missing] == [1]          # array index, not sample index
    assert corrupt == []


def test_status_flags_truncated_pickle(tmp_path):
    """A task killed mid-write leaves a file that exists but cannot be read."""
    from suep_plot.status import check

    out = _fake_run(tmp_path, [("sig", 0)], written=["sig.part0.pkl"])
    data = (out / "sig.part0.pkl").read_bytes()
    (out / "sig.part0.pkl").write_bytes(data[: len(data) // 2])

    assert [t.index for t in check(str(out))[2]] == [0]      # corrupt
    assert check(str(out), verify=False)[0]                  # existence-only: "done"


def test_status_array_ranges_are_contiguous():
    from suep_plot.status import _format_ranges

    assert _format_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert _format_ranges([3]) == "3"
    assert _format_ranges([]) == ""
