"""suep-config: show, check (static part) and the warnings it prints."""

from pathlib import Path

import yaml

from suep_plot.configtool import check, show

REPO = Path(__file__).resolve().parents[1]


def _yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))
    return path


def _small_set(root):
    s = root / "configs_small"
    _yaml(s / "samples.yaml", {"sig": {"files": ["/data/sig.root"]}})
    _yaml(s / "selections.yaml", {"highmet": "events.met > 50"})
    _yaml(s / "variables.yaml", {
        "small_met": {"expression": "events.met", "bins": [10, 0, 200]},
        "small_jet_pt": {"expression": "events.Jet.pt", "bins": [4, 0, 400]}})
    _yaml(s / "histograms.yaml", {"regions": {
        "evt": {"evt_sel": ["highmet"], "evt_vars": [{"var": "small_met", "name": "met"}]},
        "jets": {"objects": {"all": {"vars": [{"var": "small_jet_pt", "name": "pt"}]}}},
    }})
    _yaml(s / "columns.yaml", {"steps": []})
    return s


def test_show_lists_resolved_histograms_and_where_they_come_from(tmp_path, capsys):
    s = _small_set(tmp_path)

    assert show(str(s)) == 0
    out = capsys.readouterr().out
    assert "jets/all/pt" in out and "met" in out
    assert "configs_small/histograms.yaml" in out
    assert "Regions" in out


def test_show_grep_narrows_the_listing(tmp_path, capsys):
    s = _small_set(tmp_path)

    show(str(s), grep="jets/")
    out = capsys.readouterr().out
    assert "jets/all/pt" in out
    assert "evt/met" not in out


def test_static_check_passes_a_good_set(tmp_path, capsys):
    assert check([str(_small_set(tmp_path))], static=True) == 0
    assert "OK" in capsys.readouterr().out


def test_static_check_fails_on_an_unknown_selection(tmp_path, capsys):
    s = _small_set(tmp_path)
    _yaml(s / "histograms.yaml", {"regions": {
        "evt": {"evt_sel": ["higmet"], "evt_vars": ["small_met"]}}})

    assert check([str(s)], static=True) == 1
    out = capsys.readouterr().out
    assert "did you mean 'highmet'" in out


def test_check_warns_when_a_variant_file_replaces_its_parent(tmp_path, capsys):
    _small_set(tmp_path)
    v = tmp_path / "configs_variant"
    _yaml(v / "set.yaml", {"extends": "../configs_small"})
    _yaml(v / "columns.yaml", {"steps": []})          # no _extends

    check([str(v)], static=True)
    assert "replaces the parent's columns.yaml" in capsys.readouterr().out


def test_the_shipped_variant_set_inherits_everything_but_columns():
    from suep_plot.histograms import load_histogram_defs
    from suep_plot.processor import load_columns_config

    variant = REPO / "configs" / "configs_mds_minpts50"
    base = REPO / "configs" / "configs_mds"

    assert load_histogram_defs(variant / "histograms.yaml") == \
        load_histogram_defs(base / "histograms.yaml")
    cols = load_columns_config(variant / "columns.yaml")
    assert cols["parameters"]["cluster_min_samples"] == 50
    assert cols["steps"] == load_columns_config(base / "columns.yaml")["steps"]
