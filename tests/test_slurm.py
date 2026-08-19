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


def test_missing_proxy_warns_instead_of_failing_silently(config_set, tmp_path,
                                                         monkeypatch, capsys):
    monkeypatch.delenv("X509_USER_PROXY", raising=False)

    _generate(config_set, tmp_path, proxy=str(tmp_path / "absent"))

    assert "does not exist" in capsys.readouterr().out


def test_qos_directive_is_emitted_only_when_asked(config_set, tmp_path):
    """--time above the default QOS's cap is rejected at submit without this."""
    assert "#SBATCH --qos=c_medium" in _generate(
        config_set, tmp_path, qos="c_medium", time_limit="20:00:00")
    assert "--qos" not in _generate(config_set, tmp_path)
