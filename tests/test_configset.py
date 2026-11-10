"""The one config-set loader, and the checks built on it."""

import pytest
import yaml

from suep_plot.checks import load_and_validate, validate_config_set
from suep_plot.configset import ConfigSet, load_config_set
from suep_plot.histdefs import Histogram, Selection


def _set(root, **files):
    """A minimal set: one sample, one selection, one per-event histogram."""
    content = {
        "samples.yaml": {"sig": {"files": ["/data/sig.root"]}},
        "selections.yaml": {"highmet": "events.met > 50"},
        "variables.yaml": {"tiny_met": {"expression": "events.met", "bins": [10, 0, 200]}},
        "histograms.yaml": {"regions": {"evt": {"evt_sel": ["highmet"],
                                                "evt_vars": ["tiny_met"]}}},
        "columns.yaml": {"steps": []},
        **files,
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, obj in content.items():
        (root / name).write_text(yaml.safe_dump(obj, sort_keys=False))
    return root


def test_a_set_loads_into_one_object(tmp_path):
    cs = load_config_set(_set(tmp_path / "s"))

    assert isinstance(cs, ConfigSet)
    assert list(cs.samples) == ["sig"]
    assert isinstance(cs.histograms["evt/tiny_met"], Histogram)
    assert isinstance(cs.selections["highmet"], Selection)
    assert cs.reweighters == {} and cs.corrections == {}
    assert cs.settings.steps == () and callable(cs.derive_fn)


def test_a_bad_reweight_map_is_fatal_for_a_run(tmp_path):
    s = _set(tmp_path / "s", **{"reweights.yaml": {"w": {"file": "missing.yaml"}}})

    with pytest.raises(SystemExit, match="ERROR in reweights.yaml.*map file not found"):
        load_config_set(s)


def test_a_check_reports_every_problem_in_one_pass(tmp_path):
    s = _set(tmp_path / "s", **{
        "reweights.yaml": {"w": {"file": "missing.yaml"}},
        "columns.yaml": {"steps": ["no_such_step"]},
        "histograms.yaml": {"regions": {"evt": {"evt_sel": ["higmet"],
                                                "evt_vars": ["tiny_met"]}}},
    })

    cs, problems = load_and_validate(s)

    assert cs is not None and cs.reweighters == {} and cs.derive_fn is None
    text = "\n".join(problems)
    assert "reweights.yaml: reweight 'w': map file not found" in text
    assert "columns.yaml" in text and "no_such_step" in text
    assert "unknown selection 'higmet' (did you mean 'highmet'?)" in text
    assert validate_config_set(s) == problems


def test_a_file_that_does_not_load_ends_the_check_there(tmp_path):
    s = _set(tmp_path / "s", **{"histograms.yaml": {"met": {"bins": 3}}})

    cs, problems = load_and_validate(s)

    assert cs is None and len(problems) == 1
    assert "inside 'regions' only" in problems[0]


def test_custom_columns_use_the_package_evaluator():
    """custom/ selects objects with the same expression evaluator as the fill."""
    import custom.helpers as helpers
    from suep_plot.fill import compile_expr

    assert helpers.compile_expr is compile_expr
    assert not hasattr(helpers, "_SAFE_BUILTINS")
