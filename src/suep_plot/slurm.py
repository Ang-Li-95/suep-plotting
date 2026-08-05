"""Slurm job submission for parallel histogram filling."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def submit_jobs(
    config_dir: str,
    output_dir: str,
    partition: str | None,
    account: str | None,
    time_limit: str,
    mem: str,
    conda_env: str,
    chunk_size: int,
    workers: int = 1,
    max_concurrent: int | None = None,
    files_per_job: int | None = None,
    dry_run: bool = False,
    samples_filter: list[str] | None = None,
):
    """Submit Slurm array jobs for parallel processing.

    Sample files are resolved once, here, and each array task is handed a
    written-out list of the files it owns.  Tasks therefore never walk the
    input directories themselves: a dataset that grows after submission cannot
    shift the shard boundaries, and re-running ``sbatch job.sh`` reprocesses
    exactly the same files.

    By default one array task processes one whole sample.  With *files_per_job*
    a sample's file list is split into shards of that many files and every
    shard becomes its own array task, writing ``<sample>.part<k>.pkl``;
    ``suep-plot`` sums the parts back into one sample at load time.
    """
    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()

    from .processor import _resolve_files, load_samples

    samples = load_samples(config_dir / "samples.yaml")

    if not samples:
        print("No samples defined in samples.yaml")
        sys.exit(1)

    if samples_filter:
        unknown = [s for s in samples_filter if s not in samples]
        if unknown:
            sys.exit(f"ERROR: unknown sample(s) {unknown}; "
                     f"available: {sorted(samples)}")
        samples = {k: v for k, v in samples.items() if k in samples_filter}

    if files_per_job is not None and files_per_job < 1:
        sys.exit("--files-per-job must be >= 1")

    work_dir = output_dir / "slurm"
    log_dir = work_dir / "logs"
    list_dir = work_dir / "filelists"
    for d in (work_dir, log_dir, list_dir):
        d.mkdir(parents=True, exist_ok=True)

    # One task per sample, or per files_per_job-sized shard of a sample.
    # Each task is (sample, part, files); part is None when unsplit.
    tasks: list[tuple[str, int | None, list[str]]] = []
    print("Resolving sample files...")
    for name, cfg in samples.items():
        files = _resolve_files(cfg.get("files") or [])
        if not files:
            print(f"  WARNING: sample '{name}' has no files, skipping")
            continue
        step = files_per_job or len(files)
        shards = [files[i:i + step] for i in range(0, len(files), step)]
        print(f"  {name}: {len(files)} file(s) -> {len(shards)} task(s)")
        for k, shard in enumerate(shards):
            tasks.append((name, k if files_per_job else None, shard))

    n_jobs = len(tasks)
    if n_jobs == 0:
        print("No tasks to submit")
        sys.exit(1)

    task_list = work_dir / "task_list.txt"
    with open(task_list, "w") as f:
        for name, task_part, files in tasks:
            suffix = "" if task_part is None else f".part{task_part}"
            file_list = list_dir / f"{name}{suffix}.txt"
            with open(file_list, "w") as lf:
                lf.write("\n".join(files) + "\n")
            # Part last: an empty field is only safe at the end (see job.sh).
            f.write(f"{name}\t{file_list}\t{'' if task_part is None else task_part}\n")

    repo_root = Path(__file__).resolve().parent.parent.parent

    array_spec = f"0-{n_jobs - 1}"
    if max_concurrent:
        array_spec += f"%{max_concurrent}"

    directives = [
        "#SBATCH --job-name=suep_plot",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={max(workers, 1)}",
        f"#SBATCH --array={array_spec}",
    ]
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

python -m suep_plot.cli_worker \\
    --config-dir "{config_dir}" \\
    --output-dir "{output_dir}" \\
    --sample "$SAMPLE" \\
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
    print(f"File lists    : {list_dir}")
    print(f"Job script    : {job_sh}")
    print(f"Merge script  : {merge_sh}")
    print(f"Conda env     : {conda_env}")
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

    res = subprocess.run([sbatch, str(job_sh)])
    if res.returncode != 0:
        sys.exit(f"sbatch failed (rc={res.returncode})")
    print(f"\nAfter all jobs finish, merge and plot with:\n  bash {merge_sh}")
