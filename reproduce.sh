#!/bin/bash
# Reproduce the MDS LLP cluster-study plots: three Slurm runs.
#
#   bash reproduce.sh <output-root> [--dry-run]
#
#   configs_mds           reco-only cluster properties, min cluster size 10
#   configs_mds_minpts50  the same with the standard MDS min cluster size 50
#   configs_mds_signal    truth level over the (mDark, T) grid: LLPs, matched
#                         clusters, efficiencies
#
# Each run goes to <output-root>/<config set>.  The script submits and returns;
# the commands to follow the runs and plot them are printed at the end.
# --dry-run writes the job scripts without submitting, to inspect them first.
# See docs/running.md for the Slurm workflow.
set -eo pipefail

if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "usage: bash reproduce.sh <output-root> [--dry-run]" >&2
    echo "  e.g. bash reproduce.sh /groups/<group>/<user>/suep_output" >&2
    exit 1
fi
# Before the cd: a relative <output-root> is relative to where this was called.
OUT_ROOT="$(realpath -m "$1")"
shift
cd "$(dirname "$0")"

# ── environment ──────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate rootmds
export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/private/.proxy}"

# ── the three runs: 10 input files per array task ────────────────────
for set in configs_mds configs_mds_minpts50 configs_mds_signal; do
    python -m suep_plot.cli submit -c "configs/$set" -o "$OUT_ROOT/$set" \
        --files-per-job 10 "$@"
done

echo
echo "Follow the runs (now and then, not in a loop):"
echo "  suep-status -o $OUT_ROOT/<config set>            # --resubmit redoes failed tasks"
echo "Once complete, plot each run (config set and figure folder come from its run_info.json):"
for set in configs_mds configs_mds_minpts50 configs_mds_signal; do
    echo "  suep-plot $OUT_ROOT/$set          # -> $OUT_ROOT/$set/plots/index.html"
done
