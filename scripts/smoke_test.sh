#!/bin/bash
# End-to-end smoke test: process one signal file through the smallest config
# set and plot it.  Proves the run -> pickle -> plot chain works after a change
# to the framework, in ~1 minute instead of the ~5 a real config set takes.
#
#   bash scripts/smoke_test.sh [output-dir]
#
# Without an argument a fresh directory is made under $SCRATCHDIR (or /tmp).
# A given directory must be new, empty, or an earlier smoke-test output: the
# test never deletes anything, and refuses to write into a directory it did
# not make.
#
# It is not a physics check -- for that, compare histograms before and after
# with a full config set (see docs/running.md).
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve the output directory against the caller's working directory,
# before anything changes it.
if [ -n "$1" ]; then
    OUT="$(realpath -m "$1")"
else
    OUT="$(mktemp -d "${SCRATCHDIR:-/tmp}/suep_smoke.XXXXXX")"
fi
MARKER="$OUT/.suep_smoke_test"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && [ ! -e "$MARKER" ]; then
    echo "ERROR: $OUT is not empty and was not made by this smoke test;" \
         "give a new or empty directory." >&2
    exit 1
fi
mkdir -p "$OUT"
touch "$MARKER"
cd "$REPO"

export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/private/.proxy}"
export MPLBACKEND=Agg

if ! python -c "import suep_plot" 2>/dev/null; then
    echo "ERROR: suep_plot is not importable; activate the environment first" \
         "(conda activate rootmds) and install with 'pip install -e .'" >&2
    exit 1
fi
if ! voms-proxy-info -file "$X509_USER_PROXY" -exists -valid 0:10 >/dev/null 2>&1; then
    echo "ERROR: no valid grid proxy at $X509_USER_PROXY; reading root:// inputs needs one." >&2
    echo "Create one with:" >&2
    echo "  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $X509_USER_PROXY" >&2
    exit 1
fi

# One file out of the sample's directory, frozen into a list -- the same
# --file-list path each Slurm array task takes.
python - "$OUT/one.txt" <<'PY'
import sys
from suep_plot.processor import load_samples, resolve_sample_files
cfg = load_samples("configs/configs_smoke/samples.yaml")["suep_mDark2_temp1"]
files = resolve_sample_files(cfg)
if not files:
    raise SystemExit("ERROR: no input files resolved")
open(sys.argv[1], "w").write(files[0] + "\n")
print(f"input: {files[0]}")
PY

python -m suep_plot.cli run -c configs/configs_smoke -o "$OUT" -s suep_mDark2_temp1 \
    --file-list "$OUT/one.txt" --chunk-size 2000 --force
python -m suep_plot.cli plot "$OUT" -o "$OUT/plots" -c configs/configs_smoke --formats png

for fig in csc_cluster_size csc_cluster_dr_muon; do
    if [ ! "$OUT/plots/$fig.png" -nt "$MARKER" ]; then
        echo "FAIL: $fig.png was not produced by this run" >&2
        exit 1
    fi
done
echo
echo "OK -- $(ls "$OUT"/*.pkl | wc -l) pickle, $(ls "$OUT"/plots/*.png | wc -l) figure(s) in $OUT"
