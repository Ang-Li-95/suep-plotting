"""Command-line interface.

Works both as installed console scripts (``suep-run`` / ``suep-plot`` /
``suep-submit``) and as a module with a subcommand:
``python -m suep_plot.cli {run,plot,submit} [options]``.
"""

from __future__ import annotations

import argparse
import os
import sys


def _auto_jobs(jobs: int | None, cap: int = 8) -> int:
    """Resolve --jobs: 0/None = auto (up to *cap* processes)."""
    if jobs:
        return max(jobs, 1)
    return min(cap, os.cpu_count() or 1)


def _parse_formats(spec: str) -> tuple[str, ...]:
    formats = tuple(f.strip().lstrip(".") for f in spec.split(",") if f.strip())
    return formats or ("png", "pdf")


def _is_config_set(path: str) -> bool:
    """A config set is a directory holding the YAML files, not a parent of them.

    ``configs/`` holds one directory per study (``configs/configs_mds``, ...),
    so the directory name alone doesn't say whether it can be used as ``-c``.
    """
    return os.path.isfile(os.path.join(path, "histograms.yaml"))


def _check_config_dir(path: str) -> str:
    """Fail with the list of available sets instead of a FileNotFoundError."""
    if _is_config_set(path):
        return path
    sets = sorted(d for d in os.listdir(path) if _is_config_set(os.path.join(path, d))) \
        if os.path.isdir(path) else []
    hint = ("\n  available sets: " + ", ".join(os.path.join(path, s) for s in sets)
            if sets else "")
    raise SystemExit(
        f"ERROR: {path} is not a config set (no histograms.yaml).{hint}")


def run(argv=None):
    """Process samples and fill histograms (one output file per sample)."""
    parser = argparse.ArgumentParser(prog="suep-run",
                                     description="Process MDSNano samples and fill histograms.")
    parser.add_argument("-c", "--config-dir", default="configs", help="Config set: directory with the six YAML files (e.g. configs/configs_mds)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for per-sample pickle files (default: output)")
    parser.add_argument("-s", "--samples", nargs="*", default=None, help="Process only these samples (default: all)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Events per chunk")
    parser.add_argument("--workers", type=int, default=1, help="Local worker processes (coffea FuturesExecutor); 1 = iterative")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Reprocess even when the output pickle is newer than configs and inputs")
    parser.add_argument("--plot", action="store_true",
                        help="Plot after processing (writes to <output-dir>/plots)")
    parser.add_argument("--lumi", type=float, default=None, help="(with --plot) luminosity [/fb] for label + MC scaling")
    parser.add_argument("--log", action="store_true", help="(with --plot) logarithmic y-axis")
    parser.add_argument("--normalize", action="store_true", help="(with --plot) normalize signal to unit area")
    parser.add_argument("--formats", default="png,pdf", help="(with --plot) comma-separated figure formats")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="(with --plot) parallel rendering processes (0 = auto)")
    args = parser.parse_args(argv)

    _check_config_dir(args.config_dir)

    from .processor import run_all
    run_all(args.config_dir, args.output_dir, args.samples, args.chunk_size,
            args.workers, force=args.force)

    if args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from .plot import plot_all
        plot_all(args.output_dir, os.path.join(args.output_dir, "plots"),
                 normalize=args.normalize, log_y=args.log, lumi=args.lumi,
                 config_dir=args.config_dir, jobs=_auto_jobs(args.jobs),
                 formats=_parse_formats(args.formats))


def plot(argv=None):
    """Plot histograms from processed results."""
    parser = argparse.ArgumentParser(
        prog="suep-plot",
        description="Plot histograms from per-sample pickle files.",
        epilog="Examples:\n"
               "  suep-plot output/                          # load all .pkl in directory\n"
               "  suep-plot output/sig.pkl output/bkg.pkl    # load specific files\n"
               "  suep-plot output/ --log --lumi 38.5        # log scale with lumi label\n"
               "  suep-plot output/ --formats png -j 8       # PNG only, 8 render processes\n"
               "  suep-plot output/ --save-root merged.root  # also export ROOT file\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="Pickle file(s) or directory containing .pkl files")
    parser.add_argument("-o", "--output-dir", default="plots", help="Output directory for figures")
    parser.add_argument("-c", "--config-dir", default=None,
                        help="Config set for derived_plots.yaml and plot-time styling "
                             "overrides (e.g. configs/configs_mds)")
    parser.add_argument("--normalize", action="store_true", help="Normalize signal histograms to unit area")
    parser.add_argument("--log", action="store_true", help="Logarithmic y-axis")
    parser.add_argument("--lumi", type=float, default=None,
                        help="Integrated luminosity [/fb]: shown in the CMS label and "
                             "used to normalize MC samples by xs * lumi * 1000 / sumw")
    parser.add_argument("--no-ratio", action="store_true",
                        help="Disable the Data/MC ratio panel")
    parser.add_argument("--formats", default="png,pdf",
                        help="Comma-separated figure formats (default: png,pdf)")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Parallel rendering processes (0 = auto, 1 = serial)")
    parser.add_argument("--save-root", default=None, metavar="FILE",
                        help="Additionally export all histograms to a ROOT file")
    args = parser.parse_args(argv)

    if args.config_dir is None and _is_config_set("configs"):
        args.config_dir = "configs"
    elif args.config_dir is not None:
        _check_config_dir(args.config_dir)

    os.environ.setdefault("MPLBACKEND", "Agg")
    from .plot import plot_all
    plot_all(args.input, args.output_dir, args.normalize, args.log, args.lumi,
             config_dir=args.config_dir, jobs=_auto_jobs(args.jobs),
             formats=_parse_formats(args.formats), ratio=not args.no_ratio,
             save_root=args.save_root)


def reweight(argv=None):
    """Derive a binned reweight map from two processed samples."""
    parser = argparse.ArgumentParser(
        prog="suep-reweight",
        description="Generate a binned reweight map = <num>/<den> of a processed "
                    "histogram, ready to use via 'file:' in a set's reweights.yaml.",
        epilog="Example:\n"
               "  suep-reweight output/ --hist ht --num data_2024 --den qcd \\\n"
               "      -o configs/configs_mds/ht_reweight.yaml\n"
               "  # then in configs/configs_mds/reweights.yaml:\n"
               "  #   ht_dataMC:\n"
               "  #     file: ht_reweight.yaml\n"
               "  #     apply_to: [background]\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="Pickle file(s) or directory containing .pkl files")
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


def status(argv=None):
    """Report which array tasks of a submitted run finished, and redo the rest."""
    parser = argparse.ArgumentParser(
        prog="suep-status",
        description="Check a Slurm run for missing or unreadable output pickles, "
                    "and optionally resubmit exactly those array tasks.",
        epilog="Examples:\n"
               "  suep-status -o output_mds_data\n"
               "  suep-status -o output_mds_data --resubmit\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory of a 'suep-submit' run")
    parser.add_argument("--resubmit", action="store_true",
                        help="Resubmit the incomplete tasks over the run's job.sh")
    parser.add_argument("--dry-run", action="store_true",
                        help="(with --resubmit) print the sbatch command, don't run it")
    parser.add_argument("--no-verify", action="store_true",
                        help="Only check that pickles exist and are non-empty; skip "
                             "reading them back (faster, misses truncated files)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="(with --resubmit) max simultaneous array tasks")
    args = parser.parse_args(argv)

    from .status import report, resubmit
    verify = not args.no_verify
    if args.resubmit:
        n = resubmit(args.output_dir, verify=verify, dry_run=args.dry_run,
                     max_concurrent=args.max_concurrent)
        return 0 if n == 0 else 1
    return 0 if not report(args.output_dir, verify) else 1


def submit(argv=None):
    """Submit processing jobs to Slurm."""
    parser = argparse.ArgumentParser(prog="suep-submit",
                                     description="Submit histogram-filling jobs to Slurm.")
    parser.add_argument("-c", "--config-dir", default="configs", help="Config set: directory with the six YAML files (e.g. configs/configs_mds)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for per-job pickles")
    parser.add_argument("-s", "--samples", nargs="*", default=None,
                        help="Submit only these samples (default: all)")
    parser.add_argument("--partition", default=None, help="Slurm partition")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--qos", default=None,
                        help="Slurm QOS; needed for --time over the default "
                             "QOS's cap (on CLIP c_short caps at 8h, "
                             "c_medium at 2 days, c_long at 14 days)")
    parser.add_argument("--time", default="04:00:00", help="Wall time per job")
    parser.add_argument("--mem", default="8000", help="Memory in MB per job")
    parser.add_argument("--conda-env", default="mds", help="Conda environment to activate in jobs")
    parser.add_argument("--proxy", default=None,
                        help="Grid proxy to bake into job.sh "
                             "(default: $X509_USER_PROXY, else ~/private/.proxy)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Events per chunk")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes per job (also sets --cpus-per-task)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Max simultaneous array tasks")
    parser.add_argument("--files-per-job", type=int, default=None,
                        help="Split each sample into array tasks of this many files "
                             "(default: one task per sample); parts are summed by suep-plot")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts without submitting")
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
        max_concurrent=args.max_concurrent,
        files_per_job=args.files_per_job,
        dry_run=args.dry_run,
    )


_COMMANDS = {"run": run, "plot": plot, "submit": submit, "status": status,
             "reweight": reweight}


def main(argv=None):
    """Subcommand dispatcher for ``python -m suep_plot.cli``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m suep_plot.cli {run,plot,submit,status,reweight} [options]")
        print("       (or use the console scripts suep-run / suep-plot / "
              "suep-submit / suep-status / suep-reweight)")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command '{cmd}'. choose from: {', '.join(_COMMANDS)}")
        return 2
    # subcommands that report a condition (suep-status) return an exit code
    rc = _COMMANDS[cmd](rest)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
