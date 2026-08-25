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

    seen = _run_args(monkeypatch, ["-c", "configs/configs_mds", "-o", str(tmp_path),
                                   "-s", "sig", "--file-list", str(listing),
                                   "--part", "3"])

    assert seen["file_list"] == ["/data/a.root", "/data/b.root"]   # blanks dropped
    assert seen["part"] == "3"
    assert seen["samples"] == ["sig"]


def test_file_range_is_parsed(monkeypatch, tmp_path):
    seen = _run_args(monkeypatch, ["-c", "configs/configs_mds", "-o", str(tmp_path),
                                   "--file-range", "0:1"])

    assert seen["file_range"] == (0, 1)


def test_a_malformed_file_range_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="START:END"):
        _run_args(monkeypatch, ["-c", "configs/configs_mds", "-o", str(tmp_path),
                                "--file-range", "nope"])


def test_file_list_needs_exactly_one_sample(monkeypatch, tmp_path):
    """It replaces one sample's files:, so it cannot apply to all of them."""
    listing = tmp_path / "files.txt"
    listing.write_text("/data/a.root\n")

    with pytest.raises(SystemExit, match="exactly one"):
        _run_args(monkeypatch, ["-c", "configs/configs_mds", "-o", str(tmp_path),
                                "--file-list", str(listing)])


def test_an_empty_file_list_is_rejected(monkeypatch, tmp_path):
    """Silently processing nothing would look like a successful empty shard."""
    listing = tmp_path / "files.txt"
    listing.write_text("\n \n")

    with pytest.raises(SystemExit, match="empty"):
        _run_args(monkeypatch, ["-c", "configs/configs_mds", "-o", str(tmp_path),
                                "-s", "sig", "--file-list", str(listing)])
