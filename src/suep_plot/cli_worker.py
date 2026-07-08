"""Worker script invoked by each Slurm array task (processes one sample,
optionally restricted to a slice of the sample's file list)."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--file-range", default=None, metavar="START:END",
                        help="Process only files [START, END) of the sample's "
                             "resolved file list")
    parser.add_argument("--part", default=None,
                        help="Shard tag: write <sample>.part<PART>.pkl")
    args = parser.parse_args()

    file_range = None
    if args.file_range:
        start, end = args.file_range.split(":")
        file_range = (int(start), int(end))

    from .processor import run_all
    run_all(args.config_dir, args.output_dir, [args.sample], args.chunk_size,
            args.workers, force=args.force, file_range=file_range,
            part=args.part)


if __name__ == "__main__":
    main()
