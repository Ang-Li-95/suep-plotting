"""Regression tests for the bugs found in the 2026-11-10 code review."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import hist
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suep_plot import checks, derived, inputs, results  # noqa: E402
from suep_plot import plot as plot_mod  # noqa: E402
from suep_plot.fill import FillError, build_histograms, fill_histograms  # noqa: E402
from suep_plot.histdefs import as_histograms, as_selections, load_selection_defs  # noqa: E402
from suep_plot.reweights import Reweighter  # noqa: E402

def totals(h):
    return float(h.values(flow=True).sum())


# ── selections ────────────────────────────────────────────────────


def test_a_selection_without_an_expression_is_a_config_error(tmp_path):
    path = tmp_path / "selections.yaml"
    path.write_text("good: 'events.a > 1'\nempty: {level: event}\n")

    with pytest.raises(SystemExit, match="selection 'empty': no 'expression'"):
        load_selection_defs(path)


def test_an_object_selection_over_another_collection_is_fatal():
    events = ak.Array({"X": ak.zip({"pt": [[1.0, 5.0], [9.0], [7.0]]}),
                       "Y": ak.zip({"pt": [[9.0], [3.0, 4.0], [1.0]]})})
    defs = {"x_pt": {"expression": "events.X.pt", "bins": 2, "lo": 0, "hi": 10,
                     "selections": ["y_hard"]}}
    sels = {"y_hard": {"expression": "events.Y.pt > 5"}}

    with pytest.raises(FillError, match=r"'x_pt'.*\['y_hard'\].*another collection"):
        fill_histograms(build_histograms(as_histograms(defs), ["s"]), as_histograms(defs),
                        as_selections(sels), events, "s")


def test_check_expressions_reports_a_mask_over_another_collection(monkeypatch):
    events = ak.Array({"X": ak.zip({"pt": [[1.0, 5.0], [9.0]]}),
                       "Y": ak.zip({"pt": [[9.0], [3.0, 4.0]]})})
    monkeypatch.setattr(checks, "open_events", lambda *a, **k: events)
    defs = {"x_bad": {"expression": "events.X.pt", "bins": 2, "lo": 0, "hi": 10,
                      "selections": ["y_hard"]},
            "x_ok": {"expression": "events.X.pt", "bins": 2, "lo": 0, "hi": 10,
                     "selections": ["x_hard"]}}
    sels = {"y_hard": {"expression": "events.Y.pt > 5"},
            "x_hard": {"expression": "events.X.pt > 4"}}

    problems = checks.check_expressions(
        ["f.root"], "Events", as_histograms(defs), as_selections(sels), {})

    assert len(problems) == 1 and "'x_bad'" in problems[0]


# ── weights ───────────────────────────────────────────────────────


def test_a_missing_or_nan_histogram_weight_counts_as_one():
    events = ak.Array({"a": [1.0, 2.0, 3.0], "w": [2.0, None, np.nan]})
    defs = {"h": {"expression": "events.a", "bins": 3, "lo": 0.5, "hi": 3.5,
                  "weight": "events.w"}}
    hists = build_histograms(as_histograms(defs), ["s"])

    fill_histograms(hists, as_histograms(defs), as_selections({}), events, "s")

    assert hists["h"].values().tolist() == [[2.0, 1.0, 1.0]]
    # ... which is what the same expression gives as a global reweight.
    assert Reweighter("r", {"expression": "events.w"}).event_weight(events).tolist() \
        == [2.0, 1.0, 1.0]


def test_a_nan_per_object_weight_counts_as_one():
    events = ak.Array({"J": ak.zip({"pt": [[1.0, 2.0], [3.0]],
                                    "w": [[np.nan, 3.0], [None]]})})
    defs = {"h": {"expression": "events.J.pt", "bins": 3, "lo": 0.5, "hi": 3.5,
                  "weight": "events.J.w"}}
    hists = build_histograms(as_histograms(defs), ["s"])

    fill_histograms(hists, as_histograms(defs), as_selections({}), events, "s")

    assert hists["h"].values().tolist() == [[1.0, 3.0, 1.0]]


@pytest.mark.parametrize("key", ["apply_to", "samples"])
def test_a_bare_string_names_one_group_or_sample_not_a_substring(key):
    rw = Reweighter("r", {"expression": "1.0", key: "background"})

    def applies(name):
        return rw.applies_to(name, name, False)

    assert applies("background")
    assert not any(applies(n) for n in ("", "back", "ground", "signal"))


def test_apply_to_must_be_names():
    with pytest.raises(ValueError, match="apply_to"):
        Reweighter("r", {"expression": "1.0", "apply_to": [1, 2]})


# ── the LLP collection ────────────────────────────────────────────


def _gen_events(rows):
    def gp(pdg, mom):
        n = len(pdg)
        return {"pdgId": pdg, "genPartIdxMother": mom, "vx": [3.0] * n,
                "vy": [4.0] * n, "vz": [0.0] * n, "pt": [10.0] * n,
                "eta": [0.1 * (i + 1) for i in range(n)],
                "phi": [0.2 * (i + 1) for i in range(n)], "mass": [2.0] * n}

    rows = [gp(*r) for r in rows]
    return ak.Array({"SUEPGenPart": ak.zip({k: [r[k] for r in rows] for k in rows[0]})})


def test_an_llp_without_stored_daughters_gets_nan_not_an_index_error():
    from custom.llp import _build_llps

    events = _gen_events([
        ([25, 999999, 1, -1], [-1, 0, 1, 1]),      # the usual two daughters
        ([25, 999999], [-1, 0]),                   # no daughter stored at all
        ([25, 999999, 1], [-1, 0, 1]),             # only one
    ])

    llp = _build_llps(events)

    assert ak.to_list(llp.decayR[0]) == [5.0]
    assert np.isnan(llp.decayR[1][0]) and np.isnan(llp.openingAngle[1][0])
    assert llp.decayR[2][0] == 5.0 and np.isnan(llp.openingAngle[2][0])
    assert np.isfinite(llp.openingAngle[0][0]) and llp.openingAngle[0][0] > 0


# ── plotting ──────────────────────────────────────────────────────


def test_restyle_drops_a_key_removed_from_the_config(tmp_path):
    (tmp_path / "variables.yaml").write_text(
        "h: {expression: events.a, bins: [3, 0, 3]}\n")
    (tmp_path / "histograms.yaml").write_text("regions: {r: {evt_vars: [h]}}\n")
    (tmp_path / "samples.yaml").write_text("s: {files: [x.root], label: S}\n")
    hist_defs = {"r/h": {"expression": "events.a", "bins": 3, "lo": 0, "hi": 3,
                         "rebin": 2, "log_y": True, "per_object": False}}
    sample_defs = {"s": {"files": ["x.root"], "label": "old", "scale": 10}}

    results.restyle(hist_defs, sample_defs, tmp_path)

    assert "rebin" not in hist_defs["r/h"] and "log_y" not in hist_defs["r/h"]
    assert hist_defs["r/h"]["per_object"] is False        # recorded by the fill
    assert sample_defs["s"] == {"files": ["x.root"], "label": "S"}


def test_the_ratio_error_includes_the_denominator():
    num, den = np.array([50.0, 0.0, 4.0]), np.array([100.0, 100.0, 0.0])

    ratio, err = derived._ratio_with_error(num, num, den, den)

    assert ratio[0] == 0.5
    assert err[0] == pytest.approx(0.5 * np.sqrt(1 / 50 + 1 / 100))
    assert (ratio[1], err[1]) == (0.0, 0.0)
    assert np.isnan(ratio[2]) and np.isnan(err[2])


def test_a_projection_follows_normalize_and_no_ratio(monkeypatch, tmp_path):
    h = hist.Hist(hist.axis.StrCategory(["s"], name="dataset", growth=True),
                  hist.axis.Regular(2, 0, 2, name="x"),
                  hist.axis.Regular(2, 0, 2, name="y"), storage=hist.storage.Weight())
    seen = {}
    monkeypatch.setattr(derived, "plot_histogram",
                        lambda *a, **k: seen.update(k))

    derived.plot_derived("p", {"type": "projection_x", "source": "h2"},
                               {"h2": h}, {}, str(tmp_path),
                               normalize=True, ratio=False)

    assert seen["normalize"] is True and seen["ratio"] is False


def test_plot_defaults_to_plots_when_the_run_dir_is_not_the_only_input(
        tmp_path, monkeypatch):
    from suep_plot import cli

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_info.json").write_text(json.dumps({"config_dir": str(tmp_path)}))
    extra = tmp_path / "extra.pkl"
    extra.write_bytes(b"")
    seen = []
    monkeypatch.setattr(plot_mod, "plot_all",
                        lambda inputs, output_dir, *a, **k: seen.append(output_dir) or 0)

    cli.plot([str(extra), str(run)])
    cli.plot([str(run)])

    assert seen == ["plots", str(run / "plots")]


# ── xrootd wildcards ──────────────────────────────────────────────


def test_an_xrootd_wildcard_matches_its_own_directory_only(monkeypatch):
    client = pytest.importorskip("XRootD.client")

    def entry(name, is_dir=False):
        return SimpleNamespace(name=name, statinfo=SimpleNamespace(flags=2 if is_dir else 0))

    tree = {"/eos/d": [entry("a_1.root"), entry("b_1.root"), entry("failed", True)],
            "/eos/d/failed": [entry("a_2.root")]}

    class FakeFS:
        def __init__(self, url):
            pass

        def dirlist(self, path, flags):
            return SimpleNamespace(ok=True, message=""), tree[path.rstrip("/")]

    monkeypatch.setattr(client, "FileSystem", FakeFS)

    assert inputs._resolve_xrootd("root://host//eos/d/a_*.root") == \
        ["root://host//eos/d/a_1.root"]
    # A directory is still walked all the way down.
    assert inputs._resolve_xrootd("root://host//eos/d") == [
        "root://host//eos/d/a_1.root", "root://host//eos/d/b_1.root",
        "root://host//eos/d/failed/a_2.root"]


# ── region keywords ───────────────────────────────────────────────


@pytest.mark.parametrize("old, new", [("events", "evt_sel"), ("per_event", "evt_vars")])
def test_a_former_region_keyword_is_refused_with_its_new_name(old, new):
    from suep_plot.histdefs import expand_regions

    with pytest.raises(SystemExit, match=f"'{old}' is now '{new}'"):
        expand_regions({"r": {old: ["x"]}}, {})


# ── mc_only variables ─────────────────────────────────────────────


def test_check_expressions_skips_mc_only_histograms_for_data(monkeypatch):
    """Data has no Pileup branch; that must not stop the run."""
    events = ak.Array({"met": [10.0, 60.0]})
    monkeypatch.setattr(checks, "open_events", lambda *a, **k: events)
    defs = as_histograms({
        "met": {"expression": "events.met", "bins": 2, "lo": 0, "hi": 100},
        "npu": {"expression": "events.Pileup.nTrueInt", "bins": 2, "lo": 0,
                "hi": 100, "mc_only": True}})

    def check(**kw):
        return checks.check_expressions(["f.root"], "Events", defs,
                                        as_selections({}), {}, **kw)

    assert check(is_data=True) == []
    as_mc = check()
    assert len(as_mc) == 1 and "'npu'" in as_mc[0]
