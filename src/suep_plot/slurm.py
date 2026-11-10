"""Slurm job submission for parallel histogram filling."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .status import JOBS_FILE, output_name, record_submission, write_fingerprint

# Wall-time caps of the CBE.next QOS levels.  The site's submit hook adds the
# partition prefix (c_short, g_medium, ...), so only the level is written.
# Without --qos a job gets `short`.
QOS_LIMITS = {"rapid": 3600, "short": 8 * 3600, "medium": 2 * 86400,
              "long": 14 * 86400}

# The highest array index Slurm accepts here (MaxArraySize = 3001).
MAX_ARRAY_INDEX = 3000

_TIME_FORMATS = "MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS"

PROXY_HELP = ("Create one with:\n"
              "  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out {path}")


# ── wall time and QOS ─────────────────────────────────────────────


def parse_slurm_time(spec) -> int:
    """Seconds in a Slurm ``--time`` value.

    Slurm's own reading, which is easy to get wrong by hand: ``MM``,
    ``MM:SS`` (so ``04:00`` is four *minutes*), ``HH:MM:SS``, ``D-HH``,
    ``D-HH:MM`` and ``D-HH:MM:SS``.
    """
    text = str(spec).strip()
    try:
        days = 0
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
            parts = [int(x) for x in text.split(":")]
            if not 1 <= len(parts) <= 3:
                raise ValueError
            hours, minutes, seconds = (parts + [0, 0])[:3]
        else:
            parts = [int(x) for x in text.split(":")]
            if len(parts) == 1:
                hours, minutes, seconds = 0, parts[0], 0
            elif len(parts) == 2:
                hours, (minutes, seconds) = 0, parts
            elif len(parts) == 3:
                hours, minutes, seconds = parts
            else:
                raise ValueError
    except ValueError:
        raise SystemExit(f"ERROR: --time '{spec}' is not a Slurm time "
                         f"({_TIME_FORMATS})") from None
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _human(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400} day(s)"
    if seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    return f"{seconds / 3600:.1f} h"


def choose_qos(time_limit, qos: str | None = None) -> str | None:
    """The QOS a job with this wall time needs, or None for the default.

    An explicit *qos* is kept, but checked against its cap so the mistake is
    reported here rather than as ``QOSMaxWallDurationPerJobLimit`` from
    sbatch.  Without one, the shortest level that allows *time_limit* is
    chosen: none (the default ``short``) up to 8 h, ``medium`` up to 2 days,
    ``long`` up to 14 days.
    """
    seconds = parse_slurm_time(time_limit)
    if qos:
        cap = QOS_LIMITS.get(qos.split("_")[-1])
        if cap is not None and seconds > cap:
            raise SystemExit(f"ERROR: --time {time_limit} is longer than the "
                             f"'{qos}' QOS allows ({_human(cap)}).")
        return qos
    for level in ("short", "medium", "long"):
        if seconds <= QOS_LIMITS[level]:
            return None if level == "short" else level
    raise SystemExit(f"ERROR: --time {time_limit} is longer than any QOS allows "
                     f"({_human(QOS_LIMITS['long'])}); split the work with "
                     "--files-per-job instead.")


# ── grid proxy ────────────────────────────────────────────────────


def proxy_path(proxy: str | None = None) -> str:
    """The proxy to use: *proxy*, else ``$X509_USER_PROXY``, else ``~/private/.proxy``.

    The last is a path on a shared filesystem, which the batch jobs can read
    (``voms-proxy-init``'s own default under ``/tmp`` is local to one node).
    """
    return proxy or os.environ.get("X509_USER_PROXY") \
        or str(Path.home() / "private" / ".proxy")


def proxy_seconds_left(path: str) -> int | None:
    """Seconds the proxy at *path* stays valid; None if it cannot be checked."""
    exe = shutil.which("voms-proxy-info")
    if exe is None:
        return None
    try:
        res = subprocess.run([exe, "-file", path, "-timeleft"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return 0
    try:
        return int(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def ensure_proxy(proxy: str | None = None, need_seconds: int = 600) -> str:
    """Stop with instructions unless a grid proxy valid for *need_seconds* exists.

    Every ``root://`` read needs it, and without this check its absence shows
    up as an xrootd traceback halfway through listing the inputs -- or, for a
    batch job, only once the job runs.  Returns the proxy path.
    """
    path = proxy_path(proxy)
    if not Path(path).exists():
        raise SystemExit(f"ERROR: no grid proxy at {path}; reading root:// "
                         f"inputs needs one.\n{PROXY_HELP.format(path=path)}")
    left = proxy_seconds_left(path)
    if left is None:
        print(f"  (voms-proxy-info not found: cannot check that {path} is valid)")
    elif left < need_seconds:
        state = "has expired" if left <= 0 else f"expires in {left // 60} min"
        raise SystemExit(f"ERROR: the grid proxy {path} {state}.\n"
                         f"{PROXY_HELP.format(path=path)}")
    return path


def _reads_xrootd(samples: dict) -> bool:
    return any("://" in str(spec)
               for cfg in samples.values() for spec in (cfg.get("files") or []))


# ── submission ────────────────────────────────────────────────────


def _files_per_job_for(n_files: list[int], max_tasks: int) -> int:
    """The smallest --files-per-job that keeps the array within *max_tasks*."""
    k = 1
    while sum(math.ceil(n / k) for n in n_files) > max_tasks:
        k += 1
    return k


def _check_unused(output_dir: Path, overwrite: bool) -> None:
    """Refuse to submit over a run that was already submitted into *output_dir*.

    A second submission would overwrite the first one's task list and job
    script, and its pickles would sit next to the new run's.  ``--overwrite``
    moves the old ``slurm/`` aside instead; pickles the old run left behind
    are then recognised as stale (older than the new submission) and ignored.
    """
    work = output_dir / "slurm"
    used = []
    if (work / JOBS_FILE).exists():
        used.append(f"slurm/{JOBS_FILE}")
    logs = work / "logs"
    if logs.is_dir() and any(logs.iterdir()):
        used.append("slurm/logs/")
    n_pkl = len(list(output_dir.glob("*.pkl"))) if output_dir.is_dir() else 0
    if n_pkl:
        used.append(f"{n_pkl} pickle(s)")
    if not used:
        return
    if not overwrite:
        raise SystemExit(
            f"ERROR: {output_dir} already holds a run ({', '.join(used)}).\n"
            "  Submit into a new -o directory, or pass --overwrite to move the "
            "old slurm/ aside (its pickles are then ignored as stale).")
    if work.exists():
        dest = output_dir / f"slurm.prev-{datetime.now():%Y%m%d-%H%M%S}"
        work.rename(dest)
        print(f"Moved the previous run's slurm/ to {dest}")


def _job_name(output_dir: Path) -> str:
    """``suep_<run dir>``: tells runs apart in squeue/sacct."""
    return "suep_" + (re.sub(r"[^A-Za-z0-9_.-]", "_", output_dir.name) or "run")


def submit_jobs(
    config_dir: str,
    output_dir: str,
    partition: str | None,
    account: str | None,
    time_limit: str,
    mem: str,
    conda_env: str,
    chunk_size: int,
    qos: str | None = None,
    proxy: str | None = None,
    workers: int = 1,
    files_per_job: int | None = None,
    dry_run: bool = False,
    samples_filter: list[str] | None = None,
    overwrite: bool = False,
):
    """Submit Slurm array jobs for parallel processing.

    Sample files are resolved once, here, and each array task is handed a
    written-out list of the files it owns.  Tasks therefore never walk the
    input directories themselves: a dataset that grows after submission cannot
    shift the shard boundaries, and re-running ``sbatch job.sh`` reprocesses
    exactly the same files.

    By default one array task processes one whole sample.  With *files_per_job*
    (or a sample's own ``files_per_job:`` key) a sample's file list is split
    into shards of that many files and every shard becomes its own array task,
    writing ``<sample>.part<k>.pkl``; ``suep-plot`` sums the parts back into
    one sample at load time.
    """
    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()

    from .inputs import load_samples, resolve_sample_files, select_samples

    samples = select_samples(load_samples(config_dir / "samples.yaml"),
                             samples_filter)
    if files_per_job is not None and files_per_job < 1:
        raise SystemExit("ERROR: --files-per-job must be >= 1")
    qos = choose_qos(time_limit, qos)

    _check_unused(output_dir, overwrite)
    if _reads_xrootd(samples):
        proxy = ensure_proxy(proxy)
        left = proxy_seconds_left(proxy)
        if left is not None and left < parse_slurm_time(time_limit):
            print(f"  WARNING: the grid proxy expires in {left // 3600} h, "
                  f"before a task's --time {time_limit} is up; tasks that start "
                  "late will fail to read root:// inputs. Renew it first.")

    work_dir = output_dir / "slurm"
    log_dir = work_dir / "logs"
    list_dir = work_dir / "filelists"

    # One task per sample, or per files_per_job-sized shard of a sample.
    # Each task is (sample, part, files); part is None when unsplit.
    tasks: list[tuple[str, int | None, list[str]]] = []
    n_files = []
    print("Resolving sample files...")
    for name, cfg in samples.items():
        files = resolve_sample_files(cfg)
        if not files:
            print(f"  WARNING: sample '{name}' has no files, skipping")
            continue
        per_job = files_per_job or cfg.get("files_per_job")
        step = int(per_job) if per_job else len(files)
        shards = [files[i:i + step] for i in range(0, len(files), step)]
        n_files.append(len(files))
        print(f"  {name}: {len(files)} file(s) -> {len(shards)} task(s)")
        for k, shard in enumerate(shards):
            tasks.append((name, k if per_job else None, shard))

    n_jobs = len(tasks)
    if n_jobs == 0:
        raise SystemExit("ERROR: no tasks to submit (no sample resolved any file)")
    if n_jobs - 1 > MAX_ARRAY_INDEX:
        raise SystemExit(
            f"ERROR: {n_jobs} array tasks, but Slurm here accepts indices "
            f"0-{MAX_ARRAY_INDEX} only.  Use --files-per-job "
            f"{_files_per_job_for(n_files, MAX_ARRAY_INDEX + 1)} (or more), or "
            "submit fewer samples with -s.")

    for d in (work_dir, log_dir, list_dir):
        d.mkdir(parents=True, exist_ok=True)

    task_list = work_dir / "task_list.txt"
    with open(task_list, "w") as f:
        for name, task_part, files in tasks:
            file_list = list_dir / (output_name(name, task_part)[:-len(".pkl")] + ".txt")
            with open(file_list, "w") as lf:
                lf.write("\n".join(files) + "\n")
            # Part last: an empty field is only safe at the end (see job.sh).
            f.write(f"{name}\t{file_list}\t{'' if task_part is None else task_part}\n")

    write_fingerprint(work_dir, config_dir)

    repo_root = Path(__file__).resolve().parent.parent.parent
    proxy_file = proxy_path(proxy)
    array_spec = f"0-{n_jobs - 1}"

    directives = [
        f"#SBATCH --job-name={_job_name(output_dir)}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={max(workers, 1)}",
        f"#SBATCH --array={array_spec}",
    ]
    if qos:
        directives.append(f"#SBATCH --qos={qos}")
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if account:
        directives.append(f"#SBATCH --account={account}")

    script = f"""#!/bin/bash
{chr(10).join(directives)}

set -eo pipefail

TASK="${{SLURM_ARRAY_TASK_ID}}"
TASK_LIST="{task_list}"
# Lines are "<sample>\\t<file list>\\t<part>"; <part> is empty for a whole sample
# (it must come last: bash 'read' collapses consecutive tabs, so an empty field
# in the middle would shift the others).  The file list was written at
# submission time, so this task processes exactly those files no matter what
# the input directories look like now.
IFS=$'\\t' read -r SAMPLE FILE_LIST PART <<< "$(sed -n "$((TASK + 1))p" "$TASK_LIST")"

if [[ -z "$SAMPLE" || -z "$FILE_LIST" ]]; then
    echo "ERROR: no sample for task $TASK" >&2
    exit 1
fi

EXTRA=(--file-list "$FILE_LIST")
if [[ -n "$PART" ]]; then
    EXTRA+=(--part "$PART")
fi
echo "==> task $TASK: sample=$SAMPLE part=${{PART:-all}} files=$(wc -l < "$FILE_LIST") on $(hostname) at $(date)"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {conda_env}

cd "{repo_root}"
export PYTHONPATH="{repo_root}/src:$PYTHONPATH"
# Baked in, not inherited: Slurm --export=ALL hands the submitting
# shell's proxy to the first submit, but a later resubmit from a clean
# shell would leave the workers with no grid auth and every xrootd open
# would fail with "Operation not permitted".
export X509_USER_PROXY="${{X509_USER_PROXY:-{proxy_file}}}"

python -m suep_plot.cli run \\
    --config-dir "{config_dir}" \\
    --output-dir "{output_dir}" \\
    --samples "$SAMPLE" \\
    --chunk-size {chunk_size} \\
    --workers {max(workers, 1)} \\
    --force \\
    "${{EXTRA[@]}}"

echo "==> task $TASK done at $(date)"
"""

    job_sh = work_dir / "job.sh"
    with open(job_sh, "w") as f:
        f.write(script)
    os.chmod(job_sh, 0o755)

    merge_script = f"""#!/bin/bash
# Plot from per-sample histograms
eval "$(conda shell.bash hook)"
conda activate {conda_env}
cd "{repo_root}"
export PYTHONPATH="{repo_root}/src:$PYTHONPATH"
export MPLBACKEND=Agg
python -m suep_plot.cli plot "{output_dir}" -o "{output_dir}/plots" -c "{config_dir}" "$@"
"""
    merge_sh = work_dir / "merge_and_plot.sh"
    with open(merge_sh, "w") as f:
        f.write(merge_script)
    os.chmod(merge_sh, 0o755)

    print("=" * 64)
    print(f"Array tasks   : {n_jobs}")
    for i, (name, task_part, files) in enumerate(tasks):
        part_str = "" if task_part is None else f" part {task_part}"
        print(f"  [{i}] {name}{part_str} ({len(files)} files)")
    print(f"Config dir    : {config_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Work dir      : {work_dir}")
    print(f"Job script    : {job_sh}")
    print(f"Merge script  : {merge_sh}")
    print(f"Conda env     : {conda_env}")
    print(f"Wall time/QOS : {time_limit} / {qos or 'default (short)'}")
    print("=" * 64)

    if dry_run:
        print(f"[dry-run] To submit:  sbatch {job_sh}")
        print(f"          Then merge: bash {merge_sh}")
        return

    sbatch = shutil.which("sbatch")
    if not sbatch:
        print(f"'sbatch' not found. Submit manually:\n  sbatch {job_sh}")
        print(f"After jobs finish:\n  bash {merge_sh}")
        return

    res = subprocess.run([sbatch, "--parsable", str(job_sh)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"sbatch failed (rc={res.returncode}): {res.stderr.strip()}")
    job_id = res.stdout.strip().split(";")[0]
    record_submission(work_dir, job_id, array_spec)
    print(f"Submitted array job {job_id} ({n_jobs} task(s)).")
    print(f"Check progress with:  suep-status -o {output_dir}")
    print(f"After all jobs finish, merge and plot with:\n  bash {merge_sh}")
