"""Config-dir resolution: ``-c`` names one config set, never their container."""

import pytest

from suep_plot.cli import _check_config_dir, _is_config_set


def _make_set(path):
    path.mkdir(parents=True)
    (path / "histograms.yaml").write_text("{}\n")
    return path


def test_config_set_is_recognized_by_its_histograms(tmp_path):
    assert _is_config_set(str(_make_set(tmp_path / "configs_mds")))
    assert not _is_config_set(str(tmp_path))


def test_container_directory_is_rejected_and_lists_its_sets(tmp_path):
    _make_set(tmp_path / "configs" / "configs_mds")
    _make_set(tmp_path / "configs" / "configs_mds_gen")

    with pytest.raises(SystemExit) as excinfo:
        _check_config_dir(str(tmp_path / "configs"))

    message = str(excinfo.value)
    assert "not a config set" in message
    assert "configs_mds" in message and "configs_mds_gen" in message


def test_missing_directory_is_rejected_without_a_set_list(tmp_path):
    with pytest.raises(SystemExit, match="not a config set"):
        _check_config_dir(str(tmp_path / "nope"))


def test_config_set_passes_through(tmp_path):
    cfg = _make_set(tmp_path / "configs_mds")
    assert _check_config_dir(str(cfg)) == str(cfg)
