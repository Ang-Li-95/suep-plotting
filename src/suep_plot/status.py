"""Completion check for a submitted Slurm run, and resubmission of what failed.

An array task is "done" when its output pickle exists, can be read back and
was written by *this* submission.  Existence alone is not enough: a task
killed mid-write leaves a truncated pickle behind, and a pickle left over from
an earlier submission into the same directory would otherwise be counted as
this run's output and summed into the merge.

The task list written at submission time (``<outdir>/slurm/task_list.txt``) is
the authority on what *should* be there -- one line per array index, in index
order -- so the missing indices map straight onto an ``sbatch --array`` list
over the original ``job.sh``.  Resubmission therefore reprocesses exactly the
files the failed tasks owned, with the same shard boundaries.

Every ``sbatch`` of the run is recorded in ``slurm/jobs.tsv``, so one ``sacct``
call can tell a task that failed (and why) from one that is still pending or
running.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

JOBS_FILE = "jobs.tsv"
FINGERPRINT_FILE = "fingerprint.json"

# Slurm states in which a task is still in the queue or running: resubmitting
# it would run the same shard twice.
LIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING",
               "CONFIGURING", "RESIZING", "SIGNALING", "STAGE_OUT"}

# A pickle written this many seconds before the recorded submission time is
# still treated as this submission's (clock granularity), anything older as a
# leftover from an earlier run.
_CLOCK_SLACK = 2.0


def output_name(sample: str, part=None) -> str:
    """The pickle one task writes: ``<sample>.pkl`` or ``<sample>.part<part>.pkl``.

    The single definition of the naming, shared by ``suep-run`` (which writes
    the file), ``suep-status`` (which looks for it) and ``suep-plot`` (which
    decides what to merge).
    """
    suffix = f".part{part}" if part not in (None, "") else ""
    return f"{sample}{suffix}.pkl"


class Task:
    """One array task: which sample and shard it owns, and where it writes."""

    def __init__(self, index: int, sample: str, file_list: str, part: str):
        self.index = index
        self.sample = sample
        self.file_list = file_list
        self.part = part
        self.pkl_name = output_name(sample, part)

    def nfiles(self) -> int:
        try:
            with open(self.file_list) as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0


def read_task_list(output_dir: Path) -> list[Task]:
    """Parse slurm/task_list.txt into Tasks, indexed as the array was."""
    path = Path(output_dir) / "slurm" / "task_list.txt"
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


# ── submission bookkeeping ────────────────────────────────────────


def record_submission(work_dir: Path, job_id: str, array: str, note: str = "") -> None:
    """Append one ``sbatch`` of this run to ``slurm/jobs.tsv``."""
    path = Path(work_dir) / JOBS_FILE
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write("job_id\tarray\tsubmitted\toverrides\n")
        stamp = datetime.now().isoformat(timespec="seconds")
        f.write(f"{job_id}\t{array}\t{stamp}\t{note}\n")


def read_jobs(work_dir: Path) -> list[dict]:
    """The recorded submissions, oldest first (empty for runs that predate it)."""
    path = Path(work_dir) / JOBS_FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0].strip():
            rows.append({"job_id": fields[0].strip(), "array": fields[1],
                         "submitted": fields[2],
                         "overrides": fields[3] if len(fields) > 3 else ""})
    return rows


def submission_time(work_dir: Path) -> float:
    """When this run was first submitted, as a POSIX timestamp.

    The first ``jobs.tsv`` row; for runs that predate ``jobs.tsv`` (or were
    submitted by hand after ``--dry-run``) the time ``task_list.txt`` was
    written, which is the same moment to within the submission itself.
    """
    work_dir = Path(work_dir)
    jobs = read_jobs(work_dir)
    if jobs:
        try:
            return datetime.fromisoformat(jobs[0]["submitted"]).timestamp()
        except ValueError:
            pass
    task_list = work_dir / "task_list.txt"
    return task_list.stat().st_mtime if task_list.exists() else 0.0


def submitted_job_ids(work_dir: Path) -> list[str]:
    return [j["job_id"] for j in read_jobs(work_dir)]


# ── what the scheduler says ───────────────────────────────────────


def parse_sacct(text: str, job_order: list[str]) -> dict[int, tuple[str, str]]:
    """``sacct -X --array -n -P --format=JobID,State,...`` -> {index: (state, job_id)}.

    A resubmitted index appears under several jobs; the latest submission in
    *job_order* wins.  ``State`` can read ``CANCELLED by 1234``, so only its
    first word is kept.
    """
    rank = {jid: i for i, jid in enumerate(job_order)}
    best: dict[int, tuple[str, str, int]] = {}
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2 or "_" not in fields[0]:
            continue
        jid, _, idx = fields[0].partition("_")
        if not idx.isdigit():          # a condensed "[3-9]" pending range
            continue
        state = fields[1].split()[0] if fields[1].strip() else "UNKNOWN"
        r = rank.get(jid, -1)
        prev = best.get(int(idx))
        if prev is None or r >= prev[2]:
            best[int(idx)] = (state, jid, r)
    return {i: (s, j) for i, (s, j, _) in best.items()}


def slurm_states(work_dir: Path) -> dict[int, tuple[str, str]]:
    """Latest Slurm state of every array index of the run, from one ``sacct`` call.

    Empty when the run predates ``jobs.tsv`` or ``sacct`` is unavailable --
    the completion check then falls back to the pickles alone.
    """
    ids = submitted_job_ids(work_dir)
    sacct = shutil.which("sacct")
    if not ids or not sacct:
        return {}
    cmd = [sacct, "-j", ",".join(ids), "-X", "--array", "-n", "-P",
           "--format=JobID,State,ExitCode,Elapsed"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if res.returncode != 0:
        return {}
    return parse_sacct(res.stdout, ids)


def log_path(work_dir: Path, job_id: str, index: int) -> Path | None:
    """The stderr log of one array task (``%x_%A_%a.err``), if it exists."""
    hits = sorted((Path(work_dir) / "logs").glob(f"*_{job_id}_{index}.err"))
    return hits[0] if hits else None


# ── the completion check ──────────────────────────────────────────


def _is_readable(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def classify(output_dir: str, verify: bool = True) -> tuple[dict[str, list[Task]], list[Path]]:
    """Sort the run's tasks by the state of their pickle.

    Returns ``(by_state, extra)``: *by_state* maps ``done`` / ``missing`` /
    ``stale`` / ``corrupt`` to tasks, and *extra* lists the ``*.pkl`` files in
    the directory that no task of this run writes.

    ``stale`` is a pickle older than this submission -- left over from an
    earlier run into the same directory, so not this run's output.  *verify*
    reads each pickle back; without it a truncated file is not caught.
    """
    out = Path(output_dir).resolve()
    tasks = read_task_list(out)
    since = submission_time(out / "slurm")
    by_state: dict[str, list[Task]] = {"done": [], "missing": [], "stale": [],
                                        "corrupt": []}
    expected = set()
    for t in tasks:
        expected.add(t.pkl_name)
        pkl = out / t.pkl_name
        if not pkl.exists() or pkl.stat().st_size == 0:
            by_state["missing"].append(t)
        elif pkl.stat().st_mtime + _CLOCK_SLACK < since:
            by_state["stale"].append(t)
        elif verify and not _is_readable(pkl):
            by_state["corrupt"].append(t)
        else:
            by_state["done"].append(t)
    extra = sorted(p for p in out.glob("*.pkl") if p.name not in expected)
    return by_state, extra


def check(output_dir: str, verify: bool = True) -> tuple[list[Task], list[Task], list[Task]]:
    """``(done, missing, corrupt)``; a stale pickle counts as missing."""
    by_state, _ = classify(output_dir, verify)
    missing = sorted(by_state["missing"] + by_state["stale"], key=lambda t: t.index)
    return by_state["done"], missing, by_state["corrupt"]


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


def report(output_dir: str, verify: bool = True,
           states: dict[int, tuple[str, str]] | None = None) -> list[Task]:
    """Print per-sample completion and return the tasks that are not done.

    *states* (from :func:`slurm_states`) adds what the scheduler says about
    each unfinished task: still queued or running, or failed and why, with the
    path of its log.
    """
    out = Path(output_dir).resolve()
    by_state, extra = classify(output_dir, verify)
    done = by_state["done"]
    redo = sorted(by_state["missing"] + by_state["stale"] + by_state["corrupt"],
                  key=lambda t: t.index)
    total = len(done) + len(redo)
    states = states or {}

    per = {k: _by_sample(v) for k, v in by_state.items()}
    samples = sorted({t.sample for v in by_state.values() for t in v})

    print(f"{output_dir}: {len(done)}/{total} task(s) complete")
    width = max((len(s) for s in samples), default=0)
    for s in samples:
        n_done = len(per["done"].get(s, []))
        n_all = sum(len(per[k].get(s, [])) for k in per)
        flags = [f"{len(per[k][s])} {label}"
                 for k, label in (("missing", "missing"), ("stale", "stale"),
                                  ("corrupt", "unreadable"))
                 if per[k].get(s)]
        tail = ("  <-- " + ", ".join(flags)) if flags else ""
        print(f"  {s:<{width}}  {n_done:>4}/{n_all:<4}{tail}")

    if by_state["stale"]:
        print(f"\n{len(by_state['stale'])} pickle(s) are older than this "
              "submission -- left over from an earlier run into the same "
              "directory, so they are not counted.")
    if extra:
        shown = ", ".join(p.name for p in extra[:5])
        more = f" (+{len(extra) - 5} more)" if len(extra) > 5 else ""
        print(f"\nWARNING: {len(extra)} pickle(s) here belong to no task of this "
              f"run and are ignored by suep-plot: {shown}{more}")

    if not redo:
        print("\nAll tasks complete.")
        return redo

    if states:
        _print_states(out / "slurm", redo, states)

    files = sum(t.nfiles() for t in redo)
    print(f"\n{len(redo)} task(s) not done ({files} input file(s)): "
          f"array {_format_ranges([t.index for t in redo])}")
    if by_state["corrupt"]:
        print("Unreadable pickles are overwritten by the rerun (the worker runs "
              "with --force).")
    return redo


def _print_states(work_dir: Path, redo: list[Task],
                  states: dict[int, tuple[str, str]]) -> None:
    """Group the unfinished tasks by Slurm state; show the failed ones' logs."""
    groups: dict[str, list[Task]] = {}
    for t in redo:
        state = states.get(t.index, ("NOT-IN-SACCT", ""))[0]
        groups.setdefault(state, []).append(t)
    print("\nSlurm state of the unfinished tasks:")
    for state in sorted(groups):
        tasks = groups[state]
        live = " (still in the queue or running)" if state in LIVE_STATES else ""
        print(f"  {state:<14} {len(tasks):>4}  array "
              f"{_format_ranges([t.index for t in tasks])}{live}")
        if state in LIVE_STATES or state == "NOT-IN-SACCT":
            continue
        for t in tasks[:5]:
            log = log_path(work_dir, states[t.index][1], t.index)
            where = log if log is not None else "(no log found)"
            print(f"      [{t.index}] {t.pkl_name}: {where}")
        if len(tasks) > 5:
            print(f"      ... and {len(tasks) - 5} more")


# ── consistency of a run with its config ──────────────────────────


def config_fingerprint(config_dir: str | Path) -> dict[str, str]:
    """sha256 of every file that decides what a run fills.

    The fill-relevant YAML a set reads (``_extends`` / ``_include`` /
    ``_registry`` followed) and the ``custom/`` modules.  Stored at submission
    so a resubmission can tell whether the redone tasks would fill something
    different from the tasks that already finished.
    """
    from .config import fill_sources

    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in fill_sources(config_dir) if p.exists()}


def write_fingerprint(work_dir: Path, config_dir: str | Path) -> None:
    (Path(work_dir) / FINGERPRINT_FILE).write_text(
        json.dumps(config_fingerprint(config_dir), indent=1, sort_keys=True) + "\n")


def _changed_since_submission(work_dir: Path, config_dir: Path) -> list[str] | None:
    """Files whose content changed since submission; None if no fingerprint."""
    path = Path(work_dir) / FINGERPRINT_FILE
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    new = config_fingerprint(config_dir)
    return sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))


def _job_config_dir(script: str) -> Path | None:
    """The ``--config-dir`` job.sh was written with, if it can be read off."""
    m = re.search(r'--config-dir\s+"([^"]+)"', script)
    return Path(m.group(1)) if m else None


def _stale_job_script(job_sh: Path, tasks: list[Task]) -> str | None:
    """Why *job_sh* can no longer be resubmitted as it is, or None."""
    script = job_sh.read_text()
    if "cli_worker" in script:
        return ("it calls suep_plot.cli_worker, which no longer exists (the "
                "tasks now run 'suep_plot.cli run --file-list'). Submit the "
                "run afresh with suep-submit.")
    config_dir = _job_config_dir(script)
    if config_dir is None:
        return "its --config-dir cannot be read off the script."
    if not config_dir.is_dir():
        return f"its config set {config_dir} no longer exists."
    lost = [t.file_list for t in tasks if not Path(t.file_list).exists()]
    if lost:
        return f"{len(lost)} of its frozen file lists are gone (e.g. {lost[0]})."
    return None


def _warn_columns_changed_by_mtime(job_sh: Path, config_dir: Path) -> None:
    """For runs without a fingerprint: did any columns.yaml it reads change?"""
    from .config import config_sources

    changed = [p for p in config_sources(config_dir, ("columns.yaml",))
               if p.stat().st_mtime > job_sh.stat().st_mtime]
    if changed:
        print(f"\nWARNING: {', '.join(map(str, changed))} changed after this run "
              "was submitted, so the redone tasks would use different "
              "derived-column settings than the rest of the directory.\n"
              "  Restore it, or reprocess the whole run (a fresh suep-submit).")


def resubmit(output_dir: str, verify: bool = True, dry_run: bool = False,
             overrides: dict | None = None,
             allow_config_change: bool = False) -> tuple[int, int]:
    """Resubmit the unfinished array indices over the original job.sh.

    Tasks Slurm still reports as pending or running are left alone.
    *overrides* (``time`` / ``mem`` / ``qos`` / ``partition``) go on the
    ``sbatch`` command line, where they take precedence over the script's
    ``#SBATCH`` lines -- the way to give a task that hit its time or memory
    limit more of it.  A ``time`` override above the default QOS's cap picks a
    QOS that allows it.

    Returns the number of unfinished tasks that were neither resubmitted nor
    are still in the queue (0 = everything is done or on its way).
    """
    from .slurm import choose_qos, ensure_proxy

    out = Path(output_dir).resolve()
    work = out / "slurm"
    states = slurm_states(work)
    redo = report(output_dir, verify, states)
    if not redo:
        return 0

    live = [t for t in redo if states.get(t.index, ("",))[0] in LIVE_STATES]
    todo = [t for t in redo if t not in live]
    if live:
        print(f"\n{len(live)} task(s) are still pending or running and are not "
              "resubmitted.")
    if not todo:
        return 0

    job_sh = work / "job.sh"
    if not job_sh.exists():
        sys.exit(f"ERROR: no job script at {job_sh}")
    why = _stale_job_script(job_sh, read_task_list(out))
    if why:
        sys.exit(f"ERROR: cannot resubmit over {job_sh}: {why}")

    config_dir = _job_config_dir(job_sh.read_text())
    changed = _changed_since_submission(work, config_dir)
    if changed is None:
        _warn_columns_changed_by_mtime(job_sh, config_dir)
    elif changed:
        listing = "\n".join(f"    {p}" for p in changed)
        msg = (f"these files changed since the run was submitted, so the redone "
               f"tasks would fill differently from the finished ones:\n{listing}")
        if not allow_config_change:
            sys.exit(f"ERROR: {msg}\n  Restore them, submit the run afresh, or "
                     "pass --allow-config-change if the edits cannot affect the "
                     "fills (labels, comments).")
        print(f"\nWARNING: {msg}")

    extra_args = []
    overrides = dict(overrides or {})
    if overrides.get("time"):
        extra_args.append(f"--time={overrides['time']}")
        qos = choose_qos(overrides["time"], overrides.get("qos"))
        if qos:
            extra_args.append(f"--qos={qos}")
    elif overrides.get("qos"):
        extra_args.append(f"--qos={overrides['qos']}")
    if overrides.get("mem"):
        extra_args.append(f"--mem={overrides['mem']}")
    if overrides.get("partition"):
        extra_args.append(f"--partition={overrides['partition']}")

    array = _format_ranges([t.index for t in todo])
    cmd = ["sbatch", "--parsable", f"--array={array}", *extra_args, str(job_sh)]
    print("\n  " + " ".join(cmd))
    if dry_run:
        return len(todo)
    sbatch = shutil.which("sbatch")
    if not sbatch:
        print("'sbatch' not found; submit the command above by hand.")
        return len(todo)
    if any("://" in Path(t.file_list).read_text() for t in todo):
        # The job uses $X509_USER_PROXY when the submitting shell exports it
        # (sbatch passes the environment on), else the path baked into job.sh.
        ensure_proxy(os.environ.get("X509_USER_PROXY") or job_sh_proxy(job_sh))

    res = subprocess.run([sbatch, *cmd[1:]], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"sbatch failed (rc={res.returncode}): {res.stderr.strip()}")
    job_id = res.stdout.strip().split(";")[0]
    record_submission(work, job_id, array, " ".join(extra_args))
    print(f"Submitted array job {job_id} ({len(todo)} task(s)).")
    return 0


def job_sh_proxy(job_sh: Path) -> str | None:
    """The proxy path a job script exports, if it names one."""
    m = re.search(r'X509_USER_PROXY="\$\{X509_USER_PROXY:-([^}]+)\}"',
                  job_sh.read_text())
    return m.group(1) if m else None
