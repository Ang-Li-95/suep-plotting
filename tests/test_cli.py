"""Config-dir resolution: ``-c`` names one config set, never their container."""

from pathlib import Path

import pytest

from suep_plot.cli import _check_config_dir, _is_config_set

# Absolute, so the tests pass from any working directory.
CONFIGS_MDS = str(Path(__file__).resolve().parents[1] / "configs" / "configs_mds")


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


# ── shard flags: what suep-submit's job.sh passes to suep-run ────────────────
# These used to live on a separate `python -m suep_plot.cli_worker`, so the
# Slurm path and the interactive path were two commands with two flag sets.


def _run_args(monkeypatch, argv):
    """Call cli.run(argv) with run_all stubbed; return the arguments it got."""
    from suep_plot import cli, processor

    seen = {}

    def fake_run_all(config_dir, output_dir, samples=None, chunk_size=100_000,
                     workers=1, **kwargs):
        seen.update(samples=samples, chunk_size=chunk_size, workers=workers,
                    **kwargs)

    monkeypatch.setattr(processor, "run_all", fake_run_all)
    cli.run(argv)
    return seen


def test_file_list_and_part_reach_run_all(monkeypatch, tmp_path):
    """One frozen shard of one sample -- what each Slurm array task runs."""
    listing = tmp_path / "files.txt"
    listing.write_text("/data/a.root\n\n/data/b.root\n")

    seen = _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                   "-s", "sig", "--file-list", str(listing),
                                   "--part", "3"])

    assert seen["file_list"] == ["/data/a.root", "/data/b.root"]   # blanks dropped
    assert seen["part"] == "3"
    assert seen["samples"] == ["sig"]


def test_file_range_is_parsed(monkeypatch, tmp_path):
    seen = _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                   "--file-range", "0:1", "--part", "0"])

    assert seen["file_range"] == (0, 1)
    assert seen["part"] == "0"


def test_file_range_needs_a_part_tag(monkeypatch, tmp_path):
    """Without one it would write <sample>.pkl, which reads as the whole sample."""
    with pytest.raises(SystemExit, match="needs --part"):
        _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                "--file-range", "0:1"])


def test_a_malformed_file_range_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="START:END"):
        _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                "--file-range", "nope"])


def test_file_list_needs_exactly_one_sample(monkeypatch, tmp_path):
    """It replaces one sample's files:, so it cannot apply to all of them."""
    listing = tmp_path / "files.txt"
    listing.write_text("/data/a.root\n")

    with pytest.raises(SystemExit, match="exactly one"):
        _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                "--file-list", str(listing)])


def test_an_empty_file_list_is_rejected(monkeypatch, tmp_path):
    """Silently processing nothing would look like a successful empty shard."""
    listing = tmp_path / "files.txt"
    listing.write_text("\n \n")

    with pytest.raises(SystemExit, match="empty"):
        _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path),
                                "-s", "sig", "--file-list", str(listing)])


# ── set.yaml: base sets and descriptions ──────────────────────────


def test_a_base_set_is_refused_for_running_but_not_for_plotting(tmp_path):
    base = _make_set(tmp_path / "configs" / "common")
    (base / "set.yaml").write_text("abstract: true\n")
    _make_set(tmp_path / "configs" / "configs_mds")

    with pytest.raises(SystemExit, match="base set") as excinfo:
        _check_config_dir(str(base))
    assert "configs_mds" in str(excinfo.value)
    assert _check_config_dir(str(base), runnable=False) == str(base)


def test_a_set_of_only_set_yaml_is_a_config_set(tmp_path):
    variant = tmp_path / "variant"
    variant.mkdir()
    (variant / "set.yaml").write_text("extends: ../configs_mds\n")

    assert _is_config_set(str(variant))


def test_a_misspelled_set_name_gets_a_suggestion(tmp_path):
    _make_set(tmp_path / "configs" / "configs_mds")

    with pytest.raises(SystemExit, match="did you mean"):
        _check_config_dir(str(tmp_path / "configs" / "configs_mdss"))


def test_the_set_list_shows_descriptions_and_hides_bases(tmp_path):
    study = _make_set(tmp_path / "configs" / "configs_mds")
    (study / "set.yaml").write_text("description: the reco study\n")
    base = _make_set(tmp_path / "configs" / "common")
    (base / "set.yaml").write_text("abstract: true\n")

    with pytest.raises(SystemExit) as excinfo:
        _check_config_dir(str(tmp_path / "configs"))
    message = str(excinfo.value)
    assert "the reco study" in message
    assert "common" not in message.replace("configs_mds", "")


# ── defaults, run_info.json ───────────────────────────────────────


def test_run_needs_a_config_set(capsys):
    from suep_plot import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.run(["-o", "out"])
    assert excinfo.value.code == 2
    assert "--config-dir" in capsys.readouterr().err


def test_a_local_run_records_what_produced_it(monkeypatch, tmp_path):
    import json

    _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path)])

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["config_dir"] == CONFIGS_MDS
    assert info["command"] == "suep-run"
    assert "commit" in info["git"]


def test_a_slurm_task_does_not_rewrite_run_info(monkeypatch, tmp_path):
    listing = tmp_path / "files.txt"
    listing.write_text("/data/a.root\n")

    _run_args(monkeypatch, ["-c", CONFIGS_MDS, "-o", str(tmp_path / "run"),
                            "-s", "sig", "--file-list", str(listing)])

    assert not (tmp_path / "run" / "run_info.json").exists()


def test_plot_takes_the_config_set_from_run_info(tmp_path):
    import json

    from suep_plot.cli import _plot_config_dir

    (tmp_path / "run_info.json").write_text(json.dumps({"config_dir": "/x/configs_mds"}))

    config_dir, info = _plot_config_dir([str(tmp_path)], None)
    assert config_dir == "/x/configs_mds" and info is not None
    assert _plot_config_dir([str(tmp_path)], "/y/other")[0] == "/y/other"


def test_every_command_reports_its_version(capsys):
    from suep_plot import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.status(["--version"])
    assert excinfo.value.code == 0
    assert "suep-status" in capsys.readouterr().out
