"""Command-line interface.

Works both as installed console scripts (``suep-run`` / ``suep-plot`` /
``suep-submit``) and as a module with a subcommand:
``python -m suep_plot.cli {run,plot,submit} [options]``.
"""

from __future__ import annotations

import argparse
import sys


def run(argv=None):
    """Process samples and fill histograms (one output file per sample)."""
    parser = argparse.ArgumentParser(prog="suep-run",
                                     description="Process MDSNano samples and fill histograms.")
    parser.add_argument("-c", "--config-dir", default="configs", help="Directory with YAML configs (default: configs)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for per-sample pickle files (default: output)")
    parser.add_argument("-s", "--samples", nargs="*", default=None, help="Process only these samples (default: all)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Events per chunk")
    parser.add_argument("--workers", type=int, default=1, help="Local worker processes (coffea FuturesExecutor); 1 = iterative")
    args = parser.parse_args(argv)

    from .processor import run_all
    run_all(args.config_dir, args.output_dir, args.samples, args.chunk_size, args.workers)


def plot(argv=None):
    """Plot histograms from processed results."""
    parser = argparse.ArgumentParser(
        prog="suep-plot",
        description="Plot histograms from per-sample pickle files.",
        epilog="Examples:\n"
               "  suep-plot output/                          # load all .pkl in directory\n"
               "  suep-plot output/sig.pkl output/bkg.pkl    # load specific files\n"
               "  suep-plot output/ --log --lumi 38.5        # log scale with lumi label\n"
               "  suep-plot output/ -c configs               # include derived plots\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="Pickle file(s) or directory containing .pkl files")
    parser.add_argument("-o", "--output-dir", default="plots", help="Output directory for figures")
    parser.add_argument("-c", "--config-dir", default=None,
                        help="Config directory (needed for derived_plots.yaml)")
    parser.add_argument("--normalize", action="store_true", help="Normalize signal histograms to unit area")
    parser.add_argument("--log", action="store_true", help="Logarithmic y-axis")
    parser.add_argument("--lumi", type=float, default=None,
                        help="Integrated luminosity [/fb]: shown in the CMS label and "
                             "used to normalize MC samples by xs * lumi * 1000 / sumw")
    args = parser.parse_args(argv)

    from .plot import plot_all
    plot_all(args.input, args.output_dir, args.normalize, args.log, args.lumi,
             config_dir=args.config_dir)


def submit(argv=None):
    """Submit processing jobs to Slurm."""
    parser = argparse.ArgumentParser(prog="suep-submit",
                                     description="Submit histogram-filling jobs to Slurm.")
    parser.add_argument("-c", "--config-dir", default="configs", help="Directory with YAML configs")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for per-job pickles")
    parser.add_argument("--partition", default=None, help="Slurm partition")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--time", default="04:00:00", help="Wall time per job")
    parser.add_argument("--mem", default="8000", help="Memory in MB per job")
    parser.add_argument("--conda-env", default="mds", help="Conda environment to activate in jobs")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Events per chunk")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes per job (also sets --cpus-per-task)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Max simultaneous array tasks")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts without submitting")
    args = parser.parse_args(argv)

    from .slurm import submit_jobs
    submit_jobs(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        partition=args.partition,
        account=args.account,
        time_limit=args.time,
        mem=args.mem,
        conda_env=args.conda_env,
        chunk_size=args.chunk_size,
        workers=args.workers,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
    )


_COMMANDS = {"run": run, "plot": plot, "submit": submit}


def main(argv=None):
    """Subcommand dispatcher for ``python -m suep_plot.cli``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m suep_plot.cli {run,plot,submit} [options]")
        print("       (or use the console scripts suep-run / suep-plot / suep-submit)")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command '{cmd}'. choose from: {', '.join(_COMMANDS)}")
        return 2
    _COMMANDS[cmd](rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
