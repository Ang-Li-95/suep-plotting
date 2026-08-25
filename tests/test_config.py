"""The shared config directives: _extends, _include, _registry, _repeat."""

import os

import pytest
import yaml

from suep_plot.config import ConfigError, load_config_file


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path


def test_a_missing_file_loads_as_empty(tmp_path):
    assert load_config_file(tmp_path / "absent.yaml") == {}


def test_directive_keys_never_become_definitions(tmp_path):
    _write(tmp_path / "base" / "h.yaml", {"a": {"bins": 1}})
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../base", "b": {"bins": 2}})

    assert set(load_config_file(path)) == {"a", "b"}


# ── _extends ──────────────────────────────────────────────────────


def test_extends_merges_over_the_inherited_definitions(tmp_path):
    _write(tmp_path / "base" / "h.yaml", {"keep": {"bins": 1}, "tune": {"bins": 1, "lo": 0}})
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../base", "tune": {"bins": 9}})

    # Nested mappings merge deeply, so 'lo' survives an override of 'bins'.
    assert load_config_file(path) == {"keep": {"bins": 1}, "tune": {"bins": 9, "lo": 0}}


def test_extends_null_deletes_an_inherited_definition(tmp_path):
    _write(tmp_path / "base" / "h.yaml", {"keep": {"bins": 1}, "drop": {"bins": 2}})
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../base", "drop": None})

    assert load_config_file(path) == {"keep": {"bins": 1}}


def test_extends_chains(tmp_path):
    _write(tmp_path / "a" / "h.yaml", {"x": {"bins": 1}})
    _write(tmp_path / "b" / "h.yaml", {"_extends": "../a", "y": {"bins": 2}})
    path = _write(tmp_path / "c" / "h.yaml", {"_extends": "../b", "z": {"bins": 3}})

    assert set(load_config_file(path)) == {"x", "y", "z"}


def test_extending_a_set_without_that_file_is_fine(tmp_path):
    (tmp_path / "base").mkdir()
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../base", "x": {"bins": 1}})

    assert load_config_file(path) == {"x": {"bins": 1}}


def test_extending_a_missing_config_set_raises(tmp_path):
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../nope"})

    with pytest.raises(ConfigError, match="nope"):
        load_config_file(path)


def test_circular_extends_raises(tmp_path):
    _write(tmp_path / "a" / "h.yaml", {"_extends": "../b"})
    _write(tmp_path / "b" / "h.yaml", {"_extends": "../a"})

    with pytest.raises(ConfigError, match="circular"):
        load_config_file(tmp_path / "a" / "h.yaml")


# ── _repeat ───────────────────────────────────────────────────────


def test_repeat_substitutes_into_keys_and_string_leaves(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{
            "over": [{"SYS": "csc", "SYSU": "CSC"}, {"SYS": "dt", "SYSU": "DT"}],
            "defs": {"llp_in_<SYS>": {"level": "object",
                                      "expression": "events.llp.in<SYSU>"}},
        }],
    })

    assert load_config_file(path) == {
        "llp_in_csc": {"level": "object", "expression": "events.llp.inCSC"},
        "llp_in_dt": {"level": "object", "expression": "events.llp.inDT"},
    }


def test_repeat_reads_a_named_axis(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_axes": {"systems": [{"SYS": "csc"}, {"SYS": "dt"}]},
        "_repeat": [{"over": "systems", "defs": {"<SYS>_hit": {"expression": "<SYS>"}}}],
    })

    assert set(load_config_file(path)) == {"csc_hit", "dt_hit"}


def test_repeat_leaves_latex_braces_alone(tmp_path):
    """The sigil is <NAME>, so a label full of LaTeX passes through untouched."""
    label = "$N_{\\mathrm{hits}}^{<SYSU>}$"
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{"over": [{"SYS": "csc", "SYSU": "CSC"}],
                     "defs": {"n_<SYS>": {"label": label}}}],
    })

    assert load_config_file(path)["n_csc"]["label"] == "$N_{\\mathrm{hits}}^{CSC}$"


def test_repeat_substitutes_inside_nested_lists(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{"over": [{"SYS": "csc"}],
                     "defs": {"h_<SYS>": {"selections": ["<SYS>_matched"]}}}],
    })

    assert load_config_file(path)["h_csc"]["selections"] == ["csc_matched"]


def test_repeat_can_cover_a_subset_of_the_systems(tmp_path):
    """No conditionals: a loop that skips RPC simply does not list it."""
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{"over": [{"SYS": "csc"}, {"SYS": "dt"}],
                     "defs": {"time_<SYS>": {"expression": "<SYS>.time"}}}],
    })

    assert set(load_config_file(path)) == {"time_csc", "time_dt"}


def test_a_whole_value_placeholder_keeps_its_type(tmp_path):
    """'hi: <HI>' must stay a number, not become the string "500"."""
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{
            "over": [{"SYS": "csc", "HI": 500, "EDGES": [0, 1, 2]},
                     {"SYS": "dt", "HI": 300, "EDGES": [0, 5]}],
            "defs": {"h_<SYS>": {"hi": "<HI>", "edges": "<EDGES>"}},
        }],
    })

    got = load_config_file(path)
    assert got["h_csc"] == {"hi": 500, "edges": [0, 1, 2]}
    assert got["h_dt"] == {"hi": 300, "edges": [0, 5]}


def test_an_embedded_placeholder_still_interpolates_as_text(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{"over": [{"SYS": "csc", "HI": 500}],
                     "defs": {"h_<SYS>": {"label": "up to <HI> GeV"}}}],
    })

    assert load_config_file(path)["h_csc"]["label"] == "up to 500 GeV"


def test_an_unknown_placeholder_raises(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_repeat": [{"over": [{"SYS": "csc"}], "defs": {"h_<SYS>": {"label": "<TYPO>"}}}],
    })

    with pytest.raises(ConfigError, match="TYPO"):
        load_config_file(path)


def test_an_unknown_axis_name_raises(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "_axes": {"systems": [{"SYS": "csc"}]},
        "_repeat": [{"over": "sistems", "defs": {"h_<SYS>": {}}}],
    })

    with pytest.raises(ConfigError, match="sistems"):
        load_config_file(path)


def test_a_repeat_colliding_with_an_explicit_definition_raises(tmp_path):
    path = _write(tmp_path / "s.yaml", {
        "h_csc": {"bins": 1},
        "_repeat": [{"over": [{"SYS": "csc"}], "defs": {"h_<SYS>": {"bins": 2}}}],
    })

    with pytest.raises(ConfigError, match="h_csc"):
        load_config_file(path)


def test_a_repeat_generated_definition_can_be_overridden_by_the_extending_set(tmp_path):
    _write(tmp_path / "base" / "h.yaml", {
        "_repeat": [{"over": [{"SYS": "csc"}, {"SYS": "dt"}],
                     "defs": {"h_<SYS>": {"bins": 10}}}],
    })
    path = _write(tmp_path / "set" / "h.yaml", {"_extends": "../base",
                                                "h_csc": {"bins": 99}, "h_dt": None})

    assert load_config_file(path) == {"h_csc": {"bins": 99}}


# ── _include / _registry ──────────────────────────────────────────


def test_include_merges_definitions_in(tmp_path):
    _write(tmp_path / "frag.yaml", {"shared": {"bins": 1}})
    path = _write(tmp_path / "h.yaml", {"_include": ["frag.yaml"], "own": {"bins": 2}})

    assert set(load_config_file(path)) == {"shared", "own"}


def test_include_and_registry_mean_opposite_things_for_an_empty_value(tmp_path):
    """_include merges (so a null deletes); _registry selects (so a null takes)."""
    _write(tmp_path / "pool.yaml", {"sig": {"files": ["/d"]}})

    merged = _write(tmp_path / "a" / "h.yaml", {"_include": "../pool.yaml", "sig": None})
    selected = _write(tmp_path / "b" / "h.yaml", {"_registry": "../pool.yaml", "sig": None})

    assert load_config_file(merged) == {}
    assert load_config_file(selected) == {"sig": {"files": ["/d"]}}


# ── which files a set is actually made of ─────────────────────────
# The up-to-date check stats these.  A set that inherits half its definitions
# is not described by the files in its own directory, and getting this wrong is
# silent: suep-run reports "up to date" and the stale pickles get plotted.


def _set(tmp_path, name, **files):
    d = tmp_path / name
    for stem, body in files.items():
        _write(d / f"{stem}.yaml", body)
    return d


def test_sources_follow_extends_include_and_registry(tmp_path):
    from suep_plot.config import config_sources

    _write(tmp_path / "pool.yaml", {"sig": {"files": ["/d"]}})
    _write(tmp_path / "frag.yaml", {"shared_cut": {"expression": "x"}})
    _set(tmp_path, "base", histograms={"h": {"bins": 1}})
    d = _set(tmp_path, "study",
             histograms={"_extends": "../base"},
             selections={"_include": "../frag.yaml"},
             samples={"_registry": "../pool.yaml", "sig": None})

    got = {p.name for p in config_sources(d)}

    assert "frag.yaml" in got, "an _included fragment is part of the set"
    assert "pool.yaml" in got, "the dataset registry is part of the set"
    assert (tmp_path / "base" / "histograms.yaml").resolve() in config_sources(d)


def test_the_freshness_check_sees_an_included_fragment(tmp_path):
    """Editing a shared cut must reprocess every set that includes it.

    This is the isolation-cut scan: the cuts live in one fragment several sets
    pull in, so if the check only looked in each set's own directory, every
    run after an edit would report "up to date" and plot the previous cut.
    """
    from suep_plot.processor import _inputs_mtime

    frag = _write(tmp_path / "frag.yaml", {"muon_sel_foriso": {"expression": "a"}})
    d = _set(tmp_path, "study", histograms={"h": {"bins": 1}},
             selections={"_include": "../frag.yaml"})

    before = _inputs_mtime([], d)
    os.utime(frag, (before + 100, before + 100))

    assert _inputs_mtime([], d) > before


def test_a_derived_plot_edit_does_not_force_reprocessing(tmp_path):
    """derived_plots.yaml is read by suep-plot only, so it is not an input."""
    from suep_plot.processor import _inputs_mtime

    d = _set(tmp_path, "study", histograms={"h": {"bins": 1}},
             derived_plots={"eff": {"type": "efficiency"}})

    before = _inputs_mtime([], d)
    plots = d / "derived_plots.yaml"
    os.utime(plots, (before + 100, before + 100))

    assert _inputs_mtime([], d) == before


# ── composing bases ───────────────────────────────────────────────


def test_extends_accepts_a_list_of_bases(tmp_path):
    """A study composes ../common with ../common_gen instead of chaining."""
    _write(tmp_path / "common" / "h.yaml", {"reco": {"bins": 1}})
    _write(tmp_path / "common_gen" / "h.yaml", {"truth": {"bins": 2}})
    path = _write(tmp_path / "study" / "h.yaml",
                  {"_extends": ["../common", "../common_gen"]})

    assert load_config_file(path) == {"reco": {"bins": 1}, "truth": {"bins": 2}}


def test_later_bases_win_over_earlier_ones(tmp_path):
    _write(tmp_path / "a" / "h.yaml", {"x": {"bins": 1, "lo": 0}})
    _write(tmp_path / "b" / "h.yaml", {"x": {"bins": 9}})
    path = _write(tmp_path / "study" / "h.yaml", {"_extends": ["../a", "../b"]})

    assert load_config_file(path) == {"x": {"bins": 9, "lo": 0}}


def test_a_study_can_drop_one_key_of_an_inherited_definition(tmp_path):
    """How a set opts out of a split: `name: {variants: null}`, not a rewrite."""
    _write(tmp_path / "common" / "h.yaml",
           {"h": {"bins": 1, "label": "x", "variants": "csc"}})
    path = _write(tmp_path / "study" / "h.yaml",
                  {"_extends": "../common", "h": {"variants": None}})

    assert load_config_file(path) == {"h": {"bins": 1, "label": "x"}}


def test_axes_are_inherited(tmp_path):
    """A base defines the systems axis once; everything extending it can loop."""
    _write(tmp_path / "common" / "s.yaml",
           {"_axes": {"systems": [{"SYS": "csc"}, {"SYS": "dt"}]}})
    path = _write(tmp_path / "study" / "s.yaml", {
        "_extends": "../common",
        "_repeat": [{"over": "systems", "defs": {"has_<SYS>": {"expression": "<SYS>"}}}],
    })

    assert set(load_config_file(path)) == {"has_csc", "has_dt"}


def test_an_inherited_axis_can_be_overridden(tmp_path):
    _write(tmp_path / "common" / "s.yaml",
           {"_axes": {"systems": [{"SYS": "csc"}, {"SYS": "dt"}]}})
    path = _write(tmp_path / "study" / "s.yaml", {
        "_extends": "../common",
        "_axes": {"systems": [{"SYS": "rpc"}]},
        "_repeat": [{"over": "systems", "defs": {"has_<SYS>": {"expression": "<SYS>"}}}],
    })

    assert set(load_config_file(path)) == {"has_rpc"}


def test_fragments_are_not_definitions(tmp_path):
    """_axes and friends must not surface as selections or samples."""
    path = _write(tmp_path / "s.yaml", {
        "_axes": {"systems": [{"SYS": "csc"}]},
        "_repeat": [{"over": "systems", "defs": {"has_<SYS>": {"expression": "x"}}}],
    })

    assert set(load_config_file(path)) == {"has_csc"}
    assert "_axes" in load_config_file(path, with_fragments=True)
