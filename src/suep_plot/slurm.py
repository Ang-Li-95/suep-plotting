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
    dry_run: bool = False,
):
    """Submit one Slurm job per sample for parallel processing."""
    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()

    with open(config_dir / "samples.yaml") as f:
        samples = yaml.safe_load(f)

    sample_names = list(samples.keys())
    n_jobs = len(sample_names)

    if n_jobs == 0:
        print("No samples defined in samples.yaml")
        sys.exit(1)

    work_dir = output_dir / "slurm"
    log_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    sample_list = work_dir / "sample_list.txt"
    with open(sample_list, "w") as f:
        for name in sample_names:
            f.write(name + "\n")

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
SAMPLE_LIST="{sample_list}"
SAMPLE=$(sed -n "$((TASK + 1))p" "$SAMPLE_LIST")

if [[ -z "$SAMPLE" ]]; then
    echo "ERROR: no sample for task $TASK" >&2
    exit 1
fi

echo "==> task $TASK: sample=$SAMPLE on $(hostname) at $(date)"

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
    --workers {max(workers, 1)}

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
python -c "
from suep_plot.plot import plot_all
plot_all('{output_dir}', '{output_dir}/plots', config_dir='{config_dir}')
"
"""
    merge_sh = work_dir / "merge_and_plot.sh"
    with open(merge_sh, "w") as f:
        f.write(merge_script)
    os.chmod(merge_sh, 0o755)

    print("=" * 64)
    print(f"Samples       : {n_jobs}")
    for i, s in enumerate(sample_names):
        print(f"  [{i}] {s}")
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
