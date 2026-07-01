"""Worker script invoked by each Slurm array task (processes one sample)."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    from .processor import run_all
    run_all(args.config_dir, args.output_dir, [args.sample], args.chunk_size, args.workers)


if __name__ == "__main__":
    main()
