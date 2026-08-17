"""Completion check for a submitted Slurm run, and resubmission of what failed.

An array task is "done" when its output pickle exists and can be read back.
Existence alone is not enough: a task killed mid-write leaves a truncated
pickle behind, which would otherwise be summed into the merge as a silently
incomplete sample.

The task list written at submission time (``<outdir>/slurm/task_list.txt``) is
the authority on what *should* be there -- one line per array index, in index
order -- so the missing indices map straight onto an ``sbatch --array`` list
over the original ``job.sh``.  Resubmission therefore reprocesses exactly the
files the failed tasks owned, with the same shard boundaries.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


class Task:
    """One array task: which sample and shard it owns, and where it writes."""

    def __init__(self, index: int, sample: str, file_list: str, part: str):
        self.index = index
        self.sample = sample
        self.file_list = file_list
        self.part = part
        suffix = f".part{part}" if part else ""
        self.pkl_name = f"{sample}{suffix}.pkl"

    def nfiles(self) -> int:
        try:
            with open(self.file_list) as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0


def read_task_list(output_dir: Path) -> list[Task]:
    """Parse slurm/task_list.txt into Tasks, indexed as the array was."""
    path = output_dir / "slurm" / "task_list.txt"
    if not path.exists():
        sys.exit(f"ERROR: no task list at {path}\n"
                 "That directory was not produced by 'suep-submit', or the "
                 "slurm/ scaffolding was removed.")
    tasks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            # "<sample>\t<file list>\t<part>"; part is empty for a whole sample
            fields = line.rstrip("\n").split("\t")
            sample, file_list = fields[0], fields[1]
            part = fields[2] if len(fields) > 2 else ""
            tasks.append(Task(i, sample, file_list, part))
    return tasks


def _is_readable(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def check(output_dir: str, verify: bool = True) -> tuple[list[Task], list[Task], list[Task]]:
    """Split the run's tasks into (done, missing, corrupt).

    *verify* reads each pickle back; without it only existence and a non-zero
    size are checked, which is faster but will not catch a truncated file.
    """
    out = Path(output_dir).resolve()
    tasks = read_task_list(out)

    done, missing, corrupt = [], [], []
    for t in tasks:
        pkl = out / t.pkl_name
        if not pkl.exists() or pkl.stat().st_size == 0:
            missing.append(t)
        elif verify and not _is_readable(pkl):
            corrupt.append(t)
        else:
            done.append(t)
    return done, missing, corrupt


def _by_sample(tasks: list[Task]) -> dict[str, list[Task]]:
    out: dict[str, list[Task]] = {}
    for t in tasks:
        out.setdefault(t.sample, []).append(t)
    return out


def _format_ranges(indices: list[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8' (sbatch --array syntax)."""
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def report(output_dir: str, verify: bool = True) -> list[Task]:
    """Print per-sample completion and return the tasks that need redoing."""
    done, missing, corrupt = check(output_dir, verify)
    total = len(done) + len(missing) + len(corrupt)
    redo = sorted(missing + corrupt, key=lambda t: t.index)

    per_done, per_missing, per_corrupt = (_by_sample(x) for x in (done, missing, corrupt))
    samples = sorted({t.sample for t in done + missing + corrupt})

    print(f"{output_dir}: {len(done)}/{total} task(s) complete")
    width = max((len(s) for s in samples), default=0)
    for s in samples:
        n_done = len(per_done.get(s, []))
        n_all = n_done + len(per_missing.get(s, [])) + len(per_corrupt.get(s, []))
        flags = []
        if per_missing.get(s):
            flags.append(f"{len(per_missing[s])} missing")
        if per_corrupt.get(s):
            flags.append(f"{len(per_corrupt[s])} unreadable")
        tail = ("  <-- " + ", ".join(flags)) if flags else ""
        print(f"  {s:<{width}}  {n_done:>4}/{n_all:<4}{tail}")

    if not redo:
        print("\nAll tasks complete.")
        return redo

    files = sum(t.nfiles() for t in redo)
    print(f"\n{len(redo)} task(s) to redo ({files} input file(s)): "
          f"array {_format_ranges([t.index for t in redo])}")
    if corrupt:
        print("Unreadable pickles are overwritten by the rerun (the worker runs "
              "with --force).")
    return redo


# Knobs that change what derive() computes; a rerun must use the values the
# rest of the run was filled with, or the directory ends up self-inconsistent.
_CLUSTER_ENV = ("MDS_CLUSTER_MIN_SAMPLES", "MDS_CLUSTER_EPS",
                "MDS_SKIP_CLUSTERING", "MDS_LLPIDX_CONVENTION")
_CLUSTER_DEFAULTS = {"MDS_CLUSTER_MIN_SAMPLES": "10", "MDS_CLUSTER_EPS": "0.4",
                     "MDS_SKIP_CLUSTERING": "0", "MDS_LLPIDX_CONVENTION": "genpart"}


def _warn_unpinned_env(job_sh: Path) -> None:
    """Warn when the rerun's clustering knobs come from the calling shell.

    job.sh inherits the submitting environment, so a rerun from a shell with
    different MDS_* values would fill the redone shards with different
    clustering than their neighbours -- invisible in the merged output.
    """
    script = job_sh.read_text()
    if any(f"export {k}" in script for k in _CLUSTER_ENV):
        return
    current = {k: os.environ.get(k, _CLUSTER_DEFAULTS[k]) for k in _CLUSTER_ENV}
    print("\nWARNING: job.sh does not pin the clustering knobs, so this rerun "
          "uses the current shell's:")
    for k, v in current.items():
        flag = "" if k in os.environ else "  (default, not exported)"
        print(f"    {k}={v}{flag}")
    print("  These must match the values the rest of the run was filled with.")


def resubmit(output_dir: str, verify: bool = True, dry_run: bool = False,
             max_concurrent: int | None = None) -> int:
    """Resubmit exactly the incomplete array indices over the original job.sh.

    Returns the number of tasks resubmitted (0 when the run is complete).
    """
    out = Path(output_dir).resolve()
    redo = report(output_dir, verify)
    if not redo:
        return 0

    job_sh = out / "slurm" / "job.sh"
    if not job_sh.exists():
        sys.exit(f"ERROR: no job script at {job_sh}")

    _warn_unpinned_env(job_sh)

    array = _format_ranges([t.index for t in redo])
    if max_concurrent:
        array += f"%{max_concurrent}"
    cmd = ["sbatch", f"--array={array}", str(job_sh)]

    if dry_run or not shutil.which("sbatch"):
        if not dry_run:
            print("\n'sbatch' not found. Submit manually:")
        print("\n  " + " ".join(cmd))
        return len(redo)

    print("\n  " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(f"sbatch failed (rc={res.returncode})")
    return len(redo)
