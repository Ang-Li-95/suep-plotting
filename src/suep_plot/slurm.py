"""Slurm job submission for parallel histogram filling."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


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
):
    """Submit Slurm array jobs for parallel processing.

    By default one array task processes one whole sample.  With
    *files_per_job* each sample's file list is split into shards of that many
    files and every shard becomes its own array task, writing
    ``<sample>.part<k>.pkl``; ``suep-plot`` sums the parts back into one
    sample at load time.
    """
    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with open(config_dir / "samples.yaml") as f:
        samples = yaml.safe_load(f)

    if not samples:
        print("No samples defined in samples.yaml")
        sys.exit(1)

    # One task per sample, or per files_per_job-sized shard of a sample.
    # Each task is (sample, part, start, end); part is None when unsplit.
    tasks: list[tuple[str, int | None, int | None, int | None]] = []
    if files_per_job:
        if files_per_job < 1:
            sys.exit("--files-per-job must be >= 1")
        from .processor import _resolve_files

        for name, cfg in samples.items():
            n_files = len(_resolve_files(cfg.get("files") or []))
            if n_files == 0:
                print(f"WARNING: sample '{name}' has no files, skipping")
                continue
            for k, start in enumerate(range(0, n_files, files_per_job)):
                tasks.append((name, k, start, min(start + files_per_job, n_files)))
    else:
        tasks = [(name, None, None, None) for name in samples]

    n_jobs = len(tasks)
    if n_jobs == 0:
        print("No tasks to submit")
        sys.exit(1)

    work_dir = output_dir / "slurm"
    log_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    task_list = work_dir / "task_list.txt"
    with open(task_list, "w") as f:
        for name, task_part, start, end in tasks:
            if task_part is None:
                f.write(name + "\n")
            else:
                f.write(f"{name}\t{task_part}\t{start}\t{end}\n")

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
# Lines are either "<sample>" (whole sample) or
# "<sample>\\t<part>\\t<start>\\t<end>" (shard of the sample's file list).
IFS=$'\\t' read -r SAMPLE PART START END <<< "$(sed -n "$((TASK + 1))p" "$TASK_LIST")"

if [[ -z "$SAMPLE" ]]; then
    echo "ERROR: no sample for task $TASK" >&2
    exit 1
fi

EXTRA=()
if [[ -n "$PART" ]]; then
    EXTRA+=(--file-range "${{START}}:${{END}}" --part "$PART")
    echo "==> task $TASK: sample=$SAMPLE part=$PART files=[$START:$END) on $(hostname) at $(date)"
else
    echo "==> task $TASK: sample=$SAMPLE on $(hostname) at $(date)"
fi

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
    for i, (name, task_part, start, end) in enumerate(tasks):
        if task_part is None:
            print(f"  [{i}] {name}")
        else:
            print(f"  [{i}] {name} part {task_part} (files {start}:{end})")
    print(f"Config dir    : {config_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Work dir      : {work_dir}")
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
