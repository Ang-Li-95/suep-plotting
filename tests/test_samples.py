"""Sample resolution: file/dir expansion and the central dataset registry."""

from pathlib import Path

import pytest
import yaml

from suep_plot.processor import _resolve_files, load_samples


def _write(path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return str(path)


def test_directory_is_walked_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    for p in ("a.root", "sub/b.root", "sub/notes.txt"):
        (tmp_path / p).touch()

    files = _resolve_files([str(tmp_path)])

    assert files == [str(tmp_path / "a.root"), str(tmp_path / "sub" / "b.root")]


def test_explicit_files_and_globs_still_work(tmp_path):
    for p in ("a.root", "b.root"):
        (tmp_path / p).touch()

    assert _resolve_files([str(tmp_path / "a.root")]) == [str(tmp_path / "a.root")]
    assert len(_resolve_files([str(tmp_path / "*.root")])) == 2
    assert _resolve_files(["root://host//store/x/nano_0.root"]) == \
        ["root://host//store/x/nano_0.root"]


def test_registry_entry_used_verbatim(tmp_path):
    _write(tmp_path / "datasets.yaml",
           {"sig": {"files": ["/data/sig"], "xs": 2.0, "color": "tab:blue"}})
    path = _write(tmp_path / "samples.yaml", {"_include": "datasets.yaml", "sig": None})

    assert load_samples(path) == {
        "sig": {"files": ["/data/sig"], "xs": 2.0, "color": "tab:blue"}}


def test_registry_entry_with_overrides_and_alias(tmp_path):
    _write(tmp_path / "datasets.yaml",
           {"sig": {"files": ["/data/sig"], "xs": 2.0, "color": "tab:blue"}})
    path = _write(tmp_path / "samples.yaml", {
        "_include": "datasets.yaml",
        "sig": {"color": "black"},
        "sig_copy": {"_from": "sig", "label": "same data, other name"},
    })

    samples = load_samples(path)

    assert samples["sig"] == {"files": ["/data/sig"], "xs": 2.0, "color": "black"}
    assert samples["sig_copy"] == {"files": ["/data/sig"], "xs": 2.0,
                                   "color": "tab:blue",
                                   "label": "same data, other name"}


def test_standalone_definition_needs_no_registry(tmp_path):
    path = _write(tmp_path / "samples.yaml", {"sig": {"files": ["/data/sig"], "xs": 1.0}})

    assert load_samples(path) == {"sig": {"files": ["/data/sig"], "xs": 1.0}}


def test_unknown_registry_entry_raises(tmp_path):
    _write(tmp_path / "datasets.yaml", {"sig": {"files": ["/data/sig"]}})
    path = _write(tmp_path / "samples.yaml", {"_include": "datasets.yaml", "typo": None})

    with pytest.raises(KeyError, match="typo"):
        load_samples(path)


def test_missing_include_raises(tmp_path):
    path = _write(tmp_path / "samples.yaml", {"_include": "nope.yaml", "sig": None})

    with pytest.raises(FileNotFoundError):
        load_samples(path)


def _fake_dataset(tmp_path, name, n):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"nano_{i}.root").touch()
    return d


def _submit(tmp_path, files_per_job=None, n_files=5):
    """Run a dry-run submission over a local fake dataset -> (out_dir, work_dir)."""
    from suep_plot.slurm import submit_jobs

    cfg_dir, out_dir = tmp_path / "cfg", tmp_path / "out"
    cfg_dir.mkdir()
    _write(cfg_dir / "samples.yaml",
           {"sig": {"files": [str(_fake_dataset(tmp_path, "sig", n_files))]}})
    submit_jobs(str(cfg_dir), str(out_dir), None, None, "01:00:00", "4G", "mds",
                chunk_size=1000, files_per_job=files_per_job, dry_run=True)
    return out_dir, out_dir / "slurm"


def test_submit_freezes_resolved_file_lists(tmp_path):
    _, work = _submit(tmp_path, files_per_job=2, n_files=5)

    lines = (work / "task_list.txt").read_text().splitlines()
    shards = [Path(line.split("\t")[1]).read_text().split() for line in lines]

    assert [len(s) for s in shards] == [2, 2, 1]
    assert sorted(f for s in shards for f in s) == \
        sorted(str(p) for p in (tmp_path / "sig").glob("*.root"))


def test_submit_file_lists_survive_dataset_growth(tmp_path):
    _, work = _submit(tmp_path, files_per_job=2, n_files=5)
    (tmp_path / "sig" / "nano_99.root").touch()  # dataset grows after submission

    shards = [Path(line.split("\t")[1]).read_text().split()
              for line in (work / "task_list.txt").read_text().splitlines()]

    assert [len(s) for s in shards] == [2, 2, 1]
    assert not any("nano_99" in f for s in shards for f in s)


@pytest.mark.parametrize("files_per_job,expected_part", [(None, ""), (2, "0")])
def test_job_script_parses_task_line(tmp_path, files_per_job, expected_part):
    """The generated job.sh must recover sample/file list/part from task 0."""
    import subprocess

    _, work = _submit(tmp_path, files_per_job=files_per_job)
    read_line = next(line for line in (work / "job.sh").read_text().splitlines()
                     if line.startswith("IFS="))
    script = (f'TASK=0\nTASK_LIST="{work / "task_list.txt"}"\n{read_line}\n'
              'echo "$SAMPLE|$(wc -l < "$FILE_LIST")|$PART"')

    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True).stdout.strip()

    n_files = files_per_job or 5
    assert out == f"sig|{n_files}|{expected_part}"


def test_submit_samples_filter(tmp_path):
    from suep_plot.slurm import submit_jobs

    cfg_dir, out_dir = tmp_path / "cfg", tmp_path / "out"
    cfg_dir.mkdir()
    _write(cfg_dir / "samples.yaml", {
        "sig": {"files": [str(_fake_dataset(tmp_path, "sig", 2))]},
        "bkg": {"files": [str(_fake_dataset(tmp_path, "bkg", 3))]},
    })
    submit_jobs(str(cfg_dir), str(out_dir), None, None, "01:00:00", "4G", "mds",
                chunk_size=1000, dry_run=True, samples_filter=["sig"])

    lines = (out_dir / "slurm" / "task_list.txt").read_text().splitlines()

    assert [line.split("\t")[0] for line in lines] == ["sig"]


def test_submit_rejects_unknown_sample(tmp_path):
    from suep_plot.slurm import submit_jobs

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _write(cfg_dir / "samples.yaml",
           {"sig": {"files": [str(_fake_dataset(tmp_path, "sig", 2))]}})

    with pytest.raises(SystemExit, match="typo"):
        submit_jobs(str(cfg_dir), str(tmp_path / "out"), None, None, "01:00:00",
                    "4G", "mds", chunk_size=1000, dry_run=True,
                    samples_filter=["typo"])
