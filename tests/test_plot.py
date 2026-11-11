"""Plot-side helpers: flow folding, xs scaling, merging, cutflow tables."""

import pickle

import hist
import numpy as np
import pytest

from suep_plot.derived import _plot_efficiency
from suep_plot.figures import _fold_flow, prep_1d
from suep_plot.results import apply_xs_scaling, merge_results, write_cutflow


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
    assert list(prep_1d(sh, {}).view().value) == [2.0, 3.0]
    assert list(prep_1d(sh, {"flow": "sum"}).view().value) == [2.0, 3.0]
    assert list(prep_1d(sh, {"flow": "none"}).view().value) == [1.0, 1.0]


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


def _age(path, seconds):
    import os
    import time

    t = time.time() - seconds
    os.utime(path, (t, t))


def test_status_does_not_count_a_pickle_from_an_earlier_submission(tmp_path):
    """Same name, but written before this run was submitted: not this run's output."""
    from suep_plot.status import check, classify

    out = _fake_run(tmp_path, [("sig", 0)], written=["sig.part0.pkl"])
    _age(out / "sig.part0.pkl", 3600)

    by_state, _ = classify(str(out))
    assert [t.index for t in by_state["stale"]] == [0]
    assert [t.index for t in check(str(out))[1]] == [0]      # to redo


def test_a_resubmit_of_a_hand_submitted_run_keeps_its_finished_pickles(tmp_path):
    """jobs.tsv first appears at the resubmit; that is not when the run started."""
    from suep_plot.status import classify, record_submission

    out = _fake_run(tmp_path, [("sig", 0), ("sig", 1)], written=["sig.part0.pkl"])
    _age(out / "slurm" / "task_list.txt", 3600)
    _age(out / "sig.part0.pkl", 1800)

    record_submission(out / "slurm", "123", "1")

    by_state, _ = classify(str(out))
    assert by_state["stale"] == []
    assert [t.index for t in by_state["done"]] == [0]


def test_status_lists_pickles_that_no_task_writes(tmp_path):
    from suep_plot.status import classify

    out = _fake_run(tmp_path, [("sig", "")], written=["sig.pkl", "other.pkl"])

    _, extra = classify(str(out))
    assert [p.name for p in extra] == ["other.pkl"]


def test_sacct_states_are_read_per_array_index():
    from suep_plot.status import parse_sacct

    text = ("100_0|COMPLETED|0:0|00:01:00\n"
            "100_1|FAILED|1:0|00:00:10\n"
            "101_1|RUNNING|0:0|00:00:05\n"          # the resubmission of index 1
            "100_[3-4]|PENDING|0:0|00:00:00\n"      # condensed range: not per index
            "100_5|CANCELLED by 1234|0:0|00:00:00\n")
    states = parse_sacct(text, ["100", "101"])

    assert states[0] == ("COMPLETED", "100")
    assert states[1] == ("RUNNING", "101")       # the latest submission wins
    assert states[5][0] == "CANCELLED"
    assert 3 not in states


def test_resubmit_refuses_a_job_script_that_calls_cli_worker(tmp_path):
    from suep_plot.status import resubmit

    out = _fake_run(tmp_path, [("sig", "")], written=[])
    (out / "slurm" / "job.sh").write_text(
        '#!/bin/bash\npython -m suep_plot.cli_worker --config-dir "/x"\n')

    with pytest.raises(SystemExit, match="cli_worker"):
        resubmit(str(out), dry_run=True)


def test_resubmit_puts_resource_overrides_on_the_sbatch_line(tmp_path, capsys):
    from suep_plot.status import resubmit

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    out = _fake_run(tmp_path / "run", [("sig", "")], written=[])
    (out / "slurm" / "job.sh").write_text(
        f'#!/bin/bash\npython -m suep_plot.cli run --config-dir "{cfg}"\n')

    unhandled = resubmit(str(out), dry_run=True,
                         overrides={"time": "20:00:00", "mem": "16000"})

    line = capsys.readouterr().out
    assert "--time=20:00:00" in line and "--qos=medium" in line
    assert "--mem=16000" in line
    assert unhandled == 1                     # dry run: nothing was submitted


# ── which pickles suep-plot merges ───────────────────────────────


def test_plot_merges_only_the_pickles_of_this_run(tmp_path, capsys):
    from pathlib import Path

    from suep_plot.results import resolve_inputs

    out = _fake_run(tmp_path, [("sig", 0), ("sig", 1)],
                    written=["sig.part0.pkl", "leftover.pkl"])

    files = resolve_inputs(str(out))

    assert [Path(f).name for f in files] == ["sig.part0.pkl"]
    printed = capsys.readouterr().out
    assert "undercounted" in printed and "leftover.pkl" in printed


def test_plot_refuses_a_whole_sample_next_to_its_parts(tmp_path):
    from suep_plot.results import resolve_inputs

    for name in ("sig.pkl", "sig.part0.pkl"):
        with open(tmp_path / name, "wb") as f:
            pickle.dump({"histograms": {}}, f)

    with pytest.raises(SystemExit, match="double counting"):
        resolve_inputs(str(tmp_path))


def test_plot_reports_an_input_that_does_not_exist(tmp_path):
    from suep_plot.results import resolve_inputs

    with pytest.raises(SystemExit, match="input not found"):
        resolve_inputs(str(tmp_path / "nope"))


def test_a_pickle_given_twice_is_merged_once(tmp_path):
    from suep_plot.results import resolve_inputs

    p = tmp_path / "sig.pkl"
    with open(p, "wb") as f:
        pickle.dump({"histograms": {}}, f)

    assert resolve_inputs([str(p), str(tmp_path)]) == [str(p)]


def test_a_truncated_pickle_is_reported_by_name(tmp_path):
    from suep_plot.results import load_results

    p = tmp_path / "sig.pkl"
    p.write_bytes(pickle.dumps({"histograms": {"a": 1}})[:5])

    with pytest.raises(SystemExit, match="sig.pkl"):
        load_results(str(p))


def test_merge_takes_per_object_from_whichever_pickle_filled_it(tmp_path):
    a = {"histograms": {}, "samples": {}, "hist_defs": {"h": {"expression": "x"}}}
    b = {"histograms": {}, "samples": {},
         "hist_defs": {"h": {"expression": "x", "per_object": True}}}
    paths = []
    for name, data in (("a.pkl", a), ("b.pkl", b)):
        with open(tmp_path / name, "wb") as f:
            pickle.dump(data, f)
        paths.append(str(tmp_path / name))

    assert merge_results(paths)["hist_defs"]["h"]["per_object"] is True


# ── region-aware derived plots, folders, gallery ──────────────────


def test_a_region_derived_plot_expands_per_variable():
    from suep_plot.derived import expand_region_derived

    defs = expand_region_derived({
        "matchfrac": {"type": "efficiency", "region": "csc", "numerator": "matched",
                      "denominator": "all", "vars": ["size", "eta"],
                      "label_y": "matched fraction"},
        "prof": {"type": "profile_x", "region": "llp", "cut": "in_csc",
                 "vars": [["eta", "nhits"]]},
        "explicit": {"type": "ratio", "numerator": "a", "denominator": "b"},
    })

    assert defs["csc/matchfrac/size"] == {
        "type": "efficiency", "numerator": "csc/matched/size",
        "denominator": "csc/all/size", "label_y": "matched fraction"}
    assert defs["csc/matchfrac/eta"]["numerator"] == "csc/matched/eta"
    assert defs["llp/prof/eta_vs_nhits"]["source"] == "llp/in_csc/eta_vs_nhits"
    assert defs["explicit"] == {"type": "ratio", "numerator": "a", "denominator": "b"}


def test_figures_of_region_histograms_go_into_folders(tmp_path):
    from suep_plot.figures import plot_histogram

    h = make_hist("sig", [10.0, 60.0])
    plot_histogram(h, {"label": "x", "annotation": "CSC · iso"}, {}, str(tmp_path),
                   "csc/iso/size", formats=("png",))

    assert (tmp_path / "csc" / "iso" / "size.png").exists()


def test_the_gallery_groups_by_folder_and_skips_old_figures(tmp_path):
    import os
    import time

    from suep_plot.gallery import write_gallery

    (tmp_path / "csc" / "all").mkdir(parents=True)
    (tmp_path / "csc" / "all" / "size.png").write_bytes(b"png")
    old = tmp_path / "gone.png"
    old.write_bytes(b"png")
    t = time.time() - 3600
    os.utime(old, (t, t))
    stale_pdf = tmp_path / "csc" / "all" / "size.pdf"
    stale_pdf.write_bytes(b"pdf")
    os.utime(stale_pdf, (t, t))

    page = (tmp_path / "index.html")
    write_gallery(str(tmp_path), formats=("png",), since=time.time() - 60)
    text = page.read_text()

    assert "csc/all/size.png" in text
    assert "gone.png" not in text                    # not written by this run
    assert 'href="csc/all/size.pdf"' not in text     # stale PDF not linked
    assert 'id="filter"' in text


def test_the_root_export_nests_region_histograms(tmp_path):
    import uproot

    from suep_plot.results import export_root

    export_root({"csc/all/size": make_hist("sig", [10.0])}, str(tmp_path / "out.root"))

    with uproot.open(tmp_path / "out.root") as f:
        assert f["csc/all/size/sig"].values().sum() == 1


def test_the_comparison_scripts_load_exactly_the_named_sample(tmp_path):
    """zerobias_2024C must not also pick up zerobias_2024C_ctrl's pickles."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import compare_style

    for name in ("zerobias_2024C.pkl", "zerobias_2024C_ctrl.pkl"):
        with open(tmp_path / name, "wb") as f:
            pickle.dump({"histograms": {}, "samples": {name[:-4]: {}}}, f)

    assert compare_style.samples_in(str(tmp_path)) == ["zerobias_2024C",
                                                       "zerobias_2024C_ctrl"]
    data = compare_style.load_sample(str(tmp_path), "zerobias_2024C")
    assert list(data["samples"]) == ["zerobias_2024C"]


def test_the_comparison_scripts_style_wrappers_reach_suep_plot(tmp_path):
    """cms_label / write_gallery wrap the suep_plot functions of the same name."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import compare_style

    fig, ax, _ = compare_style.figure()
    compare_style.cms_label(ax)
    compare_style.save(fig, str(tmp_path), "one")
    index = compare_style.write_gallery(str(tmp_path), "title", "subtitle",
                                        [("one", 0.5)], "score")
    assert Path(index).exists()


def test_normalize_draws_data_at_unit_area_too(tmp_path, monkeypatch):
    """Data used to stay at raw counts next to unit-area MC."""
    from suep_plot import figures

    h = make_hist("bkg", [10.0] * 30 + [60.0] * 10)
    h += make_hist("data", [10.0] * 3 + [60.0] * 5)
    sample_defs = {"data": {"is_data": True}, "bkg": {"group": "background"}}
    seen = {}

    def ratio_panel(rax, data_hists, bkg_hists):
        seen["ratio"] = [sum(x.sum().value for x in hs)
                         for hs in (data_hists, bkg_hists)]

    def finish(fig, ax, *args, **kwargs):
        seen["top"] = ax.get_ylim()[1]

    monkeypatch.setattr(figures, "_draw_ratio_panel", ratio_panel)
    monkeypatch.setattr(figures, "_finish", finish)
    figures.plot_histogram(h, {"label": "x"}, sample_defs, str(tmp_path), "x",
                           normalize=True, formats=("png",))

    assert seen["ratio"] == pytest.approx([1.0, 1.0])
    assert seen["top"] < 2.0           # unit-area points, not the raw counts


def test_the_event_display_takes_a_system_without_rechits():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import event_display

    hits = {"label": np.array([], dtype=int)}
    assert event_display.cluster_summary(hits, {"match_min_hits": 10}) == []


def test_resubmit_stops_when_slurm_cannot_be_asked(tmp_path, monkeypatch):
    """A failed sacct is not "nothing is pending": those tasks would run twice."""
    import subprocess

    from suep_plot import status

    out = _fake_run(tmp_path, [("sig", 0), ("sig", 1)], written=[])
    status.record_submission(out / "slurm", "777", "0-1")
    monkeypatch.setattr(status.shutil, "which", lambda name: f"/usr/bin/{name}")

    def sacct_fails(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "slurmdbd: connection refused")

    monkeypatch.setattr(status.subprocess, "run", sacct_fails)
    with pytest.raises(SystemExit, match="Nothing was resubmitted"):
        status.resubmit(str(out), dry_run=True)
    # The report alone still works, from the pickles.
    assert status.slurm_states(out / "slurm") == {}

    def sacct_hangs(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(status.subprocess, "run", sacct_hangs)
    with pytest.raises(SystemExit, match="did not answer"):
        status.resubmit(str(out), dry_run=True)
