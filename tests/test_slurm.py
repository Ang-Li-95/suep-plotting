"""The generated job.sh must stand on its own: grid auth and QOS baked in."""

import pytest

from suep_plot.slurm import submit_jobs


@pytest.fixture
def config_set(tmp_path):
    """A minimal config set with one local sample, enough to generate job.sh."""
    cfg = tmp_path / "configs_test"
    cfg.mkdir()
    (cfg / "histograms.yaml").write_text("{}\n")
    sample = tmp_path / "sample.root"
    sample.touch()
    (cfg / "samples.yaml").write_text(
        f"one:\n  files: ['{sample}']\n")
    return cfg


def _generate(cfg, tmp_path, **kwargs):
    out = tmp_path / "out"
    opts = dict(partition=None, account=None, time_limit="04:00:00", mem="8000",
                conda_env="mds", chunk_size=1000, dry_run=True)
    opts.update(kwargs)
    submit_jobs(config_dir=str(cfg), output_dir=str(out), **opts)
    return (out / "slurm" / "job.sh").read_text()


def test_proxy_is_written_into_the_script_not_inherited(config_set, tmp_path):
    """A resubmit from a shell without X509_USER_PROXY must still authenticate."""
    proxy = tmp_path / "myproxy"
    proxy.touch()

    script = _generate(config_set, tmp_path, proxy=str(proxy))

    assert f'export X509_USER_PROXY="${{X509_USER_PROXY:-{proxy}}}"' in script


def test_proxy_defaults_to_the_environment(config_set, tmp_path, monkeypatch):
    proxy = tmp_path / "envproxy"
    proxy.touch()
    monkeypatch.setenv("X509_USER_PROXY", str(proxy))

    assert str(proxy) in _generate(config_set, tmp_path)


def test_missing_proxy_stops_a_run_that_reads_xrootd(tmp_path, monkeypatch):
    """Caught at submission, not as an xrootd traceback or inside every job."""
    monkeypatch.delenv("X509_USER_PROXY", raising=False)
    cfg = tmp_path / "configs_remote"
    cfg.mkdir()
    (cfg / "histograms.yaml").write_text("{}\n")
    (cfg / "samples.yaml").write_text(
        "one:\n  files: ['root://host//store/x/nano_0.root']\n")

    with pytest.raises(SystemExit, match="no grid proxy"):
        _generate(cfg, tmp_path, proxy=str(tmp_path / "absent"))


def test_local_inputs_need_no_proxy(config_set, tmp_path, monkeypatch):
    monkeypatch.delenv("X509_USER_PROXY", raising=False)

    assert "#!/bin/bash" in _generate(config_set, tmp_path,
                                      proxy=str(tmp_path / "absent"))


def test_the_array_is_never_throttled(config_set, tmp_path):
    """No `%N` on the array directive: tasks run as wide as Slurm allows.

    There used to be a --max-concurrent flag adding `%N`; it was removed, and a
    throttle must not creep back in -- the scheduler's own limits are what
    should decide how many tasks run at once.
    """
    script = _generate(config_set, tmp_path)
    array = [ln for ln in script.splitlines() if "--array=" in ln]
    assert array, "no array directive emitted"
    assert "%" not in array[0], f"array directive is throttled: {array[0]}"


def test_qos_directive_is_emitted_only_when_asked(config_set, tmp_path):
    """--time above the default QOS's cap is rejected at submit without this."""
    assert "#SBATCH --qos=medium" in _generate(
        config_set, tmp_path, qos="medium", time_limit="20:00:00")
    assert "--qos" not in _generate(config_set, tmp_path)


def test_tasks_run_suep_run_itself(config_set, tmp_path):
    """One command for both paths: job.sh invokes the same CLI a user does.

    The Slurm tasks used to call a private `suep_plot.cli_worker` with its own
    flag set, so the shard options were unavailable interactively and the two
    entry points could drift.
    """
    script = _generate(config_set, tmp_path)

    assert "python -m suep_plot.cli run" in script
    assert "cli_worker" not in script
    # The per-task shard, frozen at submission time.
    assert '--file-list "$FILE_LIST"' in script
    assert '--part "$PART"' in script


# ── wall time and QOS ─────────────────────────────────────────────


@pytest.mark.parametrize("spec,seconds", [
    ("30", 1800),               # bare number = minutes
    ("04:00", 240),             # MM:SS -- four minutes, not four hours
    ("04:00:00", 14400),
    ("2-00", 172800),           # D-HH
    ("1-12:30", 131400),        # D-HH:MM
    ("1-00:00:10", 86410),      # D-HH:MM:SS
])
def test_wall_time_is_read_the_way_slurm_reads_it(spec, seconds):
    from suep_plot.slurm import parse_slurm_time

    assert parse_slurm_time(spec) == seconds


def test_a_malformed_wall_time_is_rejected():
    from suep_plot.slurm import parse_slurm_time

    with pytest.raises(SystemExit, match="not a Slurm time"):
        parse_slurm_time("4h")


def test_the_qos_follows_the_wall_time():
    from suep_plot.slurm import choose_qos

    assert choose_qos("08:00:00") is None            # the default, short
    assert choose_qos("20:00:00") == "medium"
    assert choose_qos("3-00:00:00") == "long"
    with pytest.raises(SystemExit, match="longer than any QOS"):
        choose_qos("15-00:00:00")


def test_an_explicit_qos_that_is_too_short_is_reported_before_sbatch():
    from suep_plot.slurm import choose_qos

    with pytest.raises(SystemExit, match="'short' QOS allows"):
        choose_qos("10:00:00", "short")
    assert choose_qos("10:00:00", "medium") == "medium"


def test_a_long_wall_time_gets_its_qos_in_the_script(config_set, tmp_path):
    assert "#SBATCH --qos=medium" in _generate(config_set, tmp_path,
                                               time_limit="20:00:00")


# ── the run directory ─────────────────────────────────────────────


def test_the_job_name_names_the_run(config_set, tmp_path):
    assert "#SBATCH --job-name=suep_out" in _generate(config_set, tmp_path)


def test_a_dry_run_can_be_repeated_into_the_same_directory(config_set, tmp_path):
    _generate(config_set, tmp_path)
    _generate(config_set, tmp_path)


def test_submitting_over_a_submitted_run_is_refused(config_set, tmp_path):
    """A second submission would overwrite the first one's bookkeeping."""
    _generate(config_set, tmp_path)
    work = tmp_path / "out" / "slurm"
    (work / "jobs.tsv").write_text("job_id\tarray\tsubmitted\toverrides\n"
                                   "123\t0-0\t2026-01-01T00:00:00\t\n")

    with pytest.raises(SystemExit, match="already holds a run"):
        _generate(config_set, tmp_path)

    _generate(config_set, tmp_path, overwrite=True)
    assert len(list((tmp_path / "out").glob("slurm.prev-*"))) == 1
    assert not (work / "jobs.tsv").exists()


def test_the_config_fingerprint_is_recorded(config_set, tmp_path):
    import json

    _generate(config_set, tmp_path)
    prints = json.loads((tmp_path / "out" / "slurm" / "fingerprint.json").read_text())

    assert any(p.endswith("samples.yaml") for p in prints)


def _many_files(tmp_path, n):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(n):
        (d / f"nano_{i:05d}.root").touch()
    return d


def test_an_array_beyond_the_index_limit_is_refused(tmp_path):
    cfg = tmp_path / "configs_big"
    cfg.mkdir()
    (cfg / "histograms.yaml").write_text("{}\n")
    (cfg / "samples.yaml").write_text(
        f"big:\n  files: ['{_many_files(tmp_path, 3002)}']\n")

    with pytest.raises(SystemExit, match=r"0-3000 only.*--files-per-job 2"):
        _generate(cfg, tmp_path, files_per_job=1)


def test_a_sample_can_carry_its_own_shard_size(tmp_path):
    cfg = tmp_path / "configs_shards"
    cfg.mkdir()
    (cfg / "histograms.yaml").write_text("{}\n")
    (cfg / "samples.yaml").write_text(
        f"sig:\n  files: ['{_many_files(tmp_path, 5)}']\n  files_per_job: 2\n")

    _generate(cfg, tmp_path)
    lines = (tmp_path / "out" / "slurm" / "task_list.txt").read_text().splitlines()

    assert [ln.split("\t")[2] for ln in lines] == ["0", "1", "2"]
