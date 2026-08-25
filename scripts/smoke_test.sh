#!/bin/bash
# End-to-end smoke test: process one signal file through the smallest config
# set and plot it.  Proves the run -> pickle -> plot chain works after a change
# to the framework, in ~1 minute instead of the ~5 a real config set takes.
#
#   bash scripts/smoke_test.sh [output-dir]
#
# It is not a physics check -- for that, compare histograms before and after
# with a full config set (see docs/configuration.md).
set -eo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/suep_smoke_$USER}"
export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/private/.proxy}"
export MPLBACKEND=Agg

if ! voms-proxy-info -exists -valid 0:10 >/dev/null 2>&1; then
    echo "ERROR: no valid grid proxy at $X509_USER_PROXY. Create one with:"
    echo "  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out \$HOME/private/.proxy"
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# One file out of the sample's directory, frozen into a list -- the same
# --file-list path each Slurm array task takes.
python - "$OUT/one.txt" <<'PY'
import sys
from suep_plot.processor import _resolve_files, load_samples
cfg = load_samples("configs/configs_smoke/samples.yaml")["suep_mDark2_temp1"]
files = _resolve_files(cfg["files"])
if not files:
    raise SystemExit("ERROR: no input files resolved")
open(sys.argv[1], "w").write(files[0] + "\n")
print(f"input: {files[0]}")
PY

suep-run  -c configs/configs_smoke -o "$OUT" -s suep_mDark2_temp1 \
          --file-list "$OUT/one.txt" --chunk-size 2000 --force
suep-plot "$OUT" -o "$OUT/plots" -c configs/configs_smoke --formats png

test -s "$OUT/plots/csc_cluster_size.png" && test -s "$OUT/plots/csc_cluster_dr_muon.png" || { echo "FAIL: no figure produced"; exit 1; }
echo
echo "OK -- $(ls "$OUT"/*.pkl | wc -l) pickle, $(ls "$OUT"/plots/*.png | wc -l) figure(s) in $OUT"
