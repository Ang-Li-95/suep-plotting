"""Command-line interface.

Installed console scripts: ``suep-run`` / ``suep-plot`` / ``suep-submit`` /
``suep-status`` / ``suep-reweight`` / ``suep-config``.  Without the install
the same commands work as
``python -m suep_plot.cli {run,plot,submit,status,reweight,config} [options]``.
Every command takes ``-h``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Written into a run's output directory by suep-run and suep-submit: which
# config set, arguments and code produced the pickles there.  suep-plot reads
# the config set back from it, so `suep-plot <run dir>` needs no -c.
RUN_INFO = "run_info.json"


class _Formatter(argparse.ArgumentDefaultsHelpFormatter,
                 argparse.RawDescriptionHelpFormatter):
    """Defaults in the help text, and examples printed as written."""


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("suep-plot")
    except Exception:  # noqa: BLE001 - not installed: still usable from the checkout
        return "unknown (not installed)"


def _parser(prog: str, description: str, epilog: str | None = None):
    parser = argparse.ArgumentParser(prog=prog, description=description,
                                     epilog=epilog, formatter_class=_Formatter)
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {_version()}")
    return parser


def _auto_jobs(jobs: int | None, cap: int = 8) -> int:
    """Resolve --jobs: 0/None = auto (up to *cap* processes)."""
    if jobs:
        return max(jobs, 1)
    return min(cap, os.cpu_count() or 1)


def _parse_formats(spec: str) -> tuple[str, ...]:
    formats = tuple(f.strip().lstrip(".") for f in spec.split(",") if f.strip())
    return formats or ("png", "pdf")


def _parse_file_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    try:
        start, end = spec.split(":")
        return int(start), int(end)
    except ValueError:
        raise SystemExit(f"ERROR: --file-range must be START:END, got '{spec}'")


def _read_file_list(path: str | None, samples: list[str] | None) -> list[str] | None:
    """The frozen file list of one shard (see ``suep-submit --files-per-job``)."""
    if not path:
        return None
    if not samples or len(samples) != 1:
        raise SystemExit("ERROR: --file-list replaces one sample's 'files:', so it "
                         "needs exactly one -s/--samples")
    with open(path) as f:
        files = [line.strip() for line in f if line.strip()]
    if not files:
        raise SystemExit(f"ERROR: file list '{path}' is empty")
    return files


# ── config sets ───────────────────────────────────────────────────


def _is_config_set(path: str) -> bool:
    """A config set is a directory holding the YAML files, not a parent of them.

    ``configs/`` holds one directory per study (``configs/configs_mds``, ...),
    so the directory name alone doesn't say whether it can be used as ``-c``.
    A set has a ``histograms.yaml``, or a ``set.yaml`` naming the set it
    inherits everything it does not write itself from.
    """
    return any(os.path.isfile(os.path.join(path, f))
               for f in ("histograms.yaml", "set.yaml"))


def _set_info(path: str) -> tuple[str, bool]:
    """(description, abstract) from a set's set.yaml."""
    from .config import read_set_file
    try:
        meta = read_set_file(path)
    except SystemExit:
        return "", False
    return str(meta.get("description") or ""), bool(meta.get("abstract"))


def _list_sets(container: str) -> str:
    """The runnable sets under *container*, one per line with its description."""
    if not os.path.isdir(container):
        return ""
    lines = []
    for d in sorted(os.listdir(container)):
        p = os.path.join(container, d)
        if not _is_config_set(p):
            continue
        desc, abstract = _set_info(p)
        if not abstract:
            lines.append(f"    {p}" + (f"  -- {desc}" if desc else ""))
    return ("\n  available sets:\n" + "\n".join(lines)) if lines else ""


def _check_config_dir(path: str, runnable: bool = True) -> str:
    """Fail with the list of available sets instead of a FileNotFoundError.

    With *runnable*, a base set (``abstract: true`` in its set.yaml) is
    refused too: it only exists to be extended.
    """
    if _is_config_set(path):
        if runnable and _set_info(path)[1]:
            raise SystemExit(
                f"ERROR: {path} is a base set (abstract: true in its set.yaml), "
                "only meant to be extended by others."
                + _list_sets(os.path.dirname(os.path.abspath(path))))
        return path
    container = path if os.path.isdir(path) else os.path.dirname(path.rstrip("/")) or "."
    hint = ""
    if not os.path.isdir(path) and os.path.isdir(container):
        import difflib
        close = difflib.get_close_matches(os.path.basename(path.rstrip("/")),
                                          os.listdir(container), n=1)
        if close:
            hint = f"\n  did you mean {os.path.join(container, close[0])}?"
    raise SystemExit(f"ERROR: {path} is not a config set.{hint}"
                     + _list_sets(container))


# ── run_info.json ─────────────────────────────────────────────────


def _git_state() -> dict:
    """Commit and dirty flag of the checkout the package runs from."""
    from .config import REPO_ROOT

    repo = str(REPO_ROOT)
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def write_run_info(output_dir: str, config_dir: str, command: str, args) -> None:
    """Record what produced a run directory (see RUN_INFO)."""
    info = {
        "config_dir": str(Path(config_dir).resolve()),
        "command": command,
        "argv": sys.argv,
        "args": {k: v for k, v in vars(args).items()},
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "git": _git_state(),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_INFO).write_text(json.dumps(info, indent=1, default=str) + "\n")


def read_run_info(run_dir: str) -> dict | None:
    path = Path(run_dir) / RUN_INFO
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ── the commands ──────────────────────────────────────────────────


def run(argv=None):
    """Process samples and fill histograms (one output file per sample)."""
    parser = _parser(
        "suep-run", "Process MDSNano samples and fill histograms: one pickle per "
        "sample in the output directory.",
        epilog="Examples:\n"
               "  suep-run -c configs/configs_mds -o output_mds -s suep_mDark2_temp1 \\\n"
               "      --file-range 0:1 --part 0          # one file, a quick check\n"
               "  suep-run -c configs/configs_mds -o output_mds --plot\n"
               "\nAnything beyond a few files belongs on Slurm: see suep-submit.")
    parser.add_argument("-c", "--config-dir", required=True,
                        help="Config set: a directory under configs/, e.g. "
                             "configs/configs_mds")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory for the per-sample pickles")
    parser.add_argument("-s", "--samples", nargs="*", default=None,
                        help="Process only these samples (space- or comma-separated; "
                             "default: all)")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Events per chunk (DBSCAN runs per chunk, so keep it small)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Local worker processes (coffea FuturesExecutor); 1 = iterative")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Reprocess even when the output pickle is newer than "
                             "configs and (local) inputs")
    parser.add_argument("--file-list", default=None, metavar="PATH",
                        help="Process exactly the files listed in PATH (one per line) "
                             "instead of resolving the sample's 'files:'; needs a "
                             "single --samples")
    parser.add_argument("--file-range", default=None, metavar="START:END",
                        help="Process only files [START, END) of the sample's "
                             "resolved file list (needs --part)")
    parser.add_argument("--part", default=None, metavar="TAG",
                        help="Shard tag: write <sample>.part<TAG>.pkl, which "
                             "suep-plot sums back into one sample")
    parser.add_argument("--plot", action="store_true",
                        help="Plot after processing (into <output-dir>/plots)")
    parser.add_argument("--lumi", type=float, default=None,
                        help="(with --plot) luminosity [/fb] for label + MC scaling")
    parser.add_argument("--log", action="store_true", help="(with --plot) logarithmic y-axis")
    parser.add_argument("--normalize", action="store_true",
                        help="(with --plot) normalize signal to unit area")
    parser.add_argument("--formats", default="png,pdf",
                        help="(with --plot) comma-separated figure formats")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="(with --plot) parallel rendering processes (0 = auto)")
    args = parser.parse_args(argv)

    _check_config_dir(args.config_dir)
    file_range = _parse_file_range(args.file_range)
    if file_range and args.part is None:
        raise SystemExit(
            "ERROR: --file-range processes part of a sample, so it needs --part "
            "<tag>: otherwise it writes <sample>.pkl, which reads as the whole "
            "sample and is overwritten (or skipped as up to date) by the next range.")
    file_list = _read_file_list(args.file_list, args.samples)
    # A Slurm array task (it gets --file-list) shares its output directory
    # with thousands of others; suep-submit wrote run_info.json once already.
    if file_list is None:
        write_run_info(args.output_dir, args.config_dir, "suep-run", args)

    from .processor import run_all
    run_all(args.config_dir, args.output_dir, args.samples, args.chunk_size,
            args.workers, force=args.force, file_range=file_range,
            part=args.part, file_list=file_list)

    if args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from .plot import plot_all
        failed = plot_all(args.output_dir, os.path.join(args.output_dir, "plots"),
                          normalize=args.normalize, log_y=args.log, lumi=args.lumi,
                          config_dir=args.config_dir, jobs=_auto_jobs(args.jobs),
                          formats=_parse_formats(args.formats))
        return 1 if failed else 0
    return 0


def _plot_config_dir(inputs: list[str], given: str | None) -> tuple[str | None, dict | None]:
    """The config set to style a plot with: -c, else the runs' run_info.json."""
    infos = [read_run_info(p) for p in inputs if os.path.isdir(p)]
    infos = [i for i in infos if i]
    if given is not None:
        return given, infos[0] if len(infos) == 1 else None
    dirs = {i.get("config_dir") for i in infos}
    if len(dirs) == 1 and None not in dirs:
        config_dir = dirs.pop()
        print(f"Config set from {RUN_INFO}: {config_dir}")
        return config_dir, infos[0] if len(infos) == 1 else None
    if len(dirs) > 1:
        print(f"WARNING: the inputs were made with different config sets "
              f"({', '.join(sorted(map(str, dirs)))}); pass -c to choose the styling")
    else:
        print("WARNING: no -c and no run_info.json: labels, colours and derived "
              "plots fall back to what the pickles carry")
    return None, None


def plot(argv=None):
    """Plot histograms from processed results."""
    parser = _parser(
        "suep-plot", "Plot histograms from per-sample pickle files.",
        epilog="Examples:\n"
               "  suep-plot output_mds                   # config set and output dir\n"
               "                                         # come from output_mds/run_info.json\n"
               "  suep-plot output_mds --formats png -j 4\n"
               "  suep-plot output_a output_b -o cmp -c configs/configs_mds\n"
               "  suep-plot output_mds --save-root merged.root\n"
               "\nExit code: 1 when no pickle was found or any figure failed.")
    parser.add_argument("input", nargs="+",
                        help="Pickle file(s) or run directories")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Figure directory (default: <run>/plots for a single "
                             "run directory with run_info.json, else ./plots)")
    parser.add_argument("-c", "--config-dir", default=None,
                        help="Config set for plot-time styling and derived plots "
                             "(default: from the run's run_info.json)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize signal histograms to unit area")
    parser.add_argument("--log", action="store_true", help="Logarithmic y-axis")
    parser.add_argument("--lumi", type=float, default=None,
                        help="Integrated luminosity [/fb]: shown in the CMS label and "
                             "used to normalize MC samples by xs * lumi * 1000 / sumw")
    parser.add_argument("--no-ratio", action="store_true",
                        help="Disable the Data/MC ratio panel")
    parser.add_argument("--formats", default="png,pdf",
                        help="Comma-separated figure formats")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Parallel rendering processes (0 = auto, 1 = serial)")
    parser.add_argument("--save-root", default=None, metavar="FILE",
                        help="Additionally export all histograms to a ROOT file")
    args = parser.parse_args(argv)

    if args.config_dir is not None:
        _check_config_dir(args.config_dir, runnable=False)
    config_dir, info = _plot_config_dir(args.input, args.config_dir)
    output_dir = args.output_dir
    if output_dir is None:
        # <run>/plots only when the one input *is* that run directory; with
        # several inputs (or a pickle first) there is no single run to put
        # the figures under.
        single_run = (info is not None and len(args.input) == 1
                      and os.path.isdir(args.input[0]))
        output_dir = os.path.join(args.input[0], "plots") if single_run else "plots"

    os.environ.setdefault("MPLBACKEND", "Agg")
    from .plot import plot_all
    failed = plot_all(args.input, output_dir, args.normalize, args.log, args.lumi,
                      config_dir=config_dir, jobs=_auto_jobs(args.jobs),
                      formats=_parse_formats(args.formats), ratio=not args.no_ratio,
                      save_root=args.save_root)
    return 1 if failed else 0


def reweight(argv=None):
    """Derive a binned reweight map from two processed samples."""
    parser = _parser(
        "suep-reweight",
        "Generate a binned reweight map = <num>/<den> of a processed histogram, "
        "ready to use via 'file:' in a set's reweights.yaml.",
        epilog="Example:\n"
               "  suep-reweight output/ --hist ht --num data_2024 --den qcd \\\n"
               "      -o configs/configs_mds/ht_reweight.yaml\n"
               "  # then in configs/configs_mds/reweights.yaml:\n"
               "  #   ht_dataMC:\n"
               "  #     file: ht_reweight.yaml\n"
               "  #     apply_to: [background]\n")
    parser.add_argument("input", nargs="+", help="Pickle file(s) or run directories")
    parser.add_argument("--hist", required=True, help="Histogram name (1D or 2D) to take the ratio of")
    parser.add_argument("--num", required=True, help="Numerator (target) sample name")
    parser.add_argument("--den", required=True, help="Denominator (source) sample name")
    parser.add_argument("-o", "--output", required=True, help="Output map YAML path")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Absolute ratio instead of shape-only (normalized) ratio")
    parser.add_argument("--no-clamp", action="store_true",
                        help="Weight 1 outside the map range instead of using edge bins")
    args = parser.parse_args(argv)

    from .reweights import make_reweight_map
    path = make_reweight_map(args.input, args.hist, args.num, args.den, args.output,
                             normalize=not args.no_normalize,
                             clamp=not args.no_clamp)
    print(f"Wrote reweight map to {path}")
    return 0


def status(argv=None):
    """Report which array tasks of a submitted run finished, and redo the rest."""
    parser = _parser(
        "suep-status",
        "Check a Slurm run: which tasks wrote their pickle, which are still queued "
        "or running, which failed and why -- and optionally resubmit the failed ones.",
        epilog="Examples:\n"
               "  suep-status -o output_mds_data\n"
               "  suep-status -o output_mds_data --resubmit\n"
               "  suep-status -o output_mds_data --resubmit --time 16:00:00 --mem 16000\n"
               "\nExit code: 0 when every task is done, running or resubmitted; 1 otherwise.\n"
               "One sacct call per invocation: run it now and then, not in a loop.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory of a 'suep-submit' run")
    parser.add_argument("--resubmit", action="store_true",
                        help="Resubmit the failed tasks over the run's job.sh "
                             "(tasks still queued or running are left alone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="(with --resubmit) print the sbatch command, don't run it")
    parser.add_argument("--time", default=None,
                        help="(with --resubmit) new wall time for the redone tasks; "
                             "a QOS that allows it is picked automatically")
    parser.add_argument("--mem", default=None,
                        help="(with --resubmit) new memory per task, e.g. 16000 or 16G")
    parser.add_argument("--qos", default=None, help="(with --resubmit) Slurm QOS")
    parser.add_argument("--partition", default=None,
                        help="(with --resubmit) Slurm partition")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="(with --resubmit) resubmit although config files "
                             "changed since submission (only for edits that cannot "
                             "change the fills, e.g. labels)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Only check that pickles exist and are non-empty; skip "
                             "reading them back (faster, misses truncated files)")
    args = parser.parse_args(argv)

    from .status import LIVE_STATES, report, resubmit, slurm_states
    verify = not args.no_verify
    if args.resubmit:
        overrides = {"time": args.time, "mem": args.mem, "qos": args.qos,
                     "partition": args.partition}
        unhandled = resubmit(args.output_dir, verify=verify, dry_run=args.dry_run,
                             overrides=overrides,
                             allow_config_change=args.allow_config_change)
        return 0 if unhandled == 0 else 1
    states = slurm_states(Path(args.output_dir).resolve() / "slurm")
    redo = report(args.output_dir, verify, states)
    waiting = [t for t in redo if states.get(t.index, ("",))[0] in LIVE_STATES]
    return 0 if len(waiting) == len(redo) else 1


def submit(argv=None):
    """Submit processing jobs to Slurm."""
    parser = _parser(
        "suep-submit", "Submit histogram-filling jobs to Slurm: one array task "
        "per sample, or per --files-per-job shard of one.",
        epilog="Examples:\n"
               "  suep-submit -c configs/configs_mds -o $SUEP_OUT/mds --files-per-job 5 --dry-run\n"
               "  suep-submit -c configs/configs_mds -o $SUEP_OUT/mds --files-per-job 5\n"
               "  suep-submit -c configs/configs_mds_cosmics -o $SUEP_OUT/cosmics \\\n"
               "      --time 24:00:00 --mem 16000         # a QOS for 24 h is picked\n"
               "\nThen: suep-status -o <output-dir>, and suep-plot <output-dir>.")
    parser.add_argument("-c", "--config-dir", required=True,
                        help="Config set: a directory under configs/, e.g. "
                             "configs/configs_mds")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory for the per-task pickles")
    parser.add_argument("-s", "--samples", nargs="*", default=None,
                        help="Submit only these samples (space- or comma-separated; "
                             "default: all)")
    parser.add_argument("--partition", default=None,
                        help="Slurm partition (default: the cluster's, c on CBE.next)")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--qos", default=None,
                        help="Slurm QOS (default: picked from --time: none up to 8 h, "
                             "medium up to 2 days, long up to 14 days)")
    parser.add_argument("--time", default="04:00:00", help="Wall time per task")
    parser.add_argument("--mem", default="8000", help="Memory per task (MB, or 16G)")
    parser.add_argument("--conda-env",
                        default=os.environ.get("CONDA_DEFAULT_ENV") or "rootmds",
                        help="Conda environment the tasks activate (default: the "
                             "one active now)")
    parser.add_argument("--proxy", default=None,
                        help="Grid proxy to bake into job.sh "
                             "(default: $X509_USER_PROXY, else ~/private/.proxy)")
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Events per chunk")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes per task (also sets --cpus-per-task)")
    parser.add_argument("--files-per-job", type=int, default=None,
                        help="Split each sample into array tasks of this many files "
                             "(default: a sample's own files_per_job, else one task "
                             "per sample); suep-plot sums the parts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the scripts without submitting")
    parser.add_argument("--overwrite", action="store_true",
                        help="Submit into an output directory that already holds a "
                             "run: its slurm/ is moved aside to slurm.prev-<time>, "
                             "and its pickles are ignored as stale")
    args = parser.parse_args(argv)

    _check_config_dir(args.config_dir)

    from .slurm import submit_jobs
    submit_jobs(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        samples_filter=args.samples,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        proxy=args.proxy,
        time_limit=args.time,
        mem=args.mem,
        conda_env=args.conda_env,
        chunk_size=args.chunk_size,
        workers=args.workers,
        files_per_job=args.files_per_job,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    write_run_info(args.output_dir, args.config_dir, "suep-submit", args)
    return 0


def config(argv=None):
    """Inspect and check config sets (``suep-config show|check|fields``)."""
    from .configtool import main as config_main
    return config_main(argv, _parser, _check_config_dir)


_COMMANDS = {"run": run, "plot": plot, "submit": submit, "status": status,
             "reweight": reweight, "config": config}


def main(argv=None):
    """Subcommand dispatcher for ``python -m suep_plot.cli``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python -m suep_plot.cli {{{','.join(_COMMANDS)}}} [options]")
        print("       (or the console scripts suep-run / suep-plot / suep-submit / "
              "suep-status / suep-reweight / suep-config)")
        print("each command takes -h")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    if argv[0] == "--version":
        print(f"suep-plot {_version()}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command '{cmd}'. choose from: {', '.join(_COMMANDS)}")
        return 2
    rc = _COMMANDS[cmd](rest)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
