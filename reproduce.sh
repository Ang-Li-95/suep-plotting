#!/bin/bash
# Reproduce the MDS LLP cluster-study plots locally (both min-cluster-size folders).
# For the Slurm path and full notes see README.md.
#
#   bash reproduce.sh
#
set -eo pipefail
cd "$(dirname "$0")"

# ── environment ──────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate mds
export X509_USER_PROXY=$HOME/private/.proxy

# Fail early without a live proxy (xrootd reads would otherwise error out).
if ! voms-proxy-info -exists -valid 0:10 >/dev/null 2>&1; then
    echo "ERROR: no valid grid proxy. Create one with:"
    echo "  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out \$HOME/private/.proxy"
    exit 1
fi

# ── clustering as set in configs/configs_mds/columns.yaml -> output_mds_min10/ ─
suep-run  -c configs/configs_mds -o output_mds_min10 --chunk-size 10000 --workers 8
suep-plot output_mds_min10 -o output_mds_min10/plots -c configs/configs_mds -j 8

# ── variant: DBSCAN min cluster size = 50 (CSC/DT) -> output_mds_min50/
# Its own config directory, but only the one line that differs: _extends pulls
# the rest from configs_mds, so the two runs cannot drift apart.
mkdir -p configs/configs_mds_min50
for f in samples histograms selections corrections reweights derived_plots; do
    printf '_extends: ../configs_mds\n' > "configs/configs_mds_min50/$f.yaml"
done
cat > configs/configs_mds_min50/columns.yaml <<'EOF'
_extends: ../configs_mds
parameters:
  cluster_min_samples: 50
EOF
suep-run  -c configs/configs_mds_min50 -o output_mds_min50 --chunk-size 10000 --workers 8
suep-plot output_mds_min50 -o output_mds_min50/plots -c configs/configs_mds_min50 -j 8

# ── truth-level plots over the (mDark, T) grid: LLPs, matched clusters, effs ──
suep-run  -c configs/configs_mds_signal -o output_mds_grid --chunk-size 10000 --workers 8
suep-plot output_mds_grid -o output_mds_grid/plots -c configs/configs_mds_signal -j 8

echo
echo "Done. Galleries:"
echo "  output_mds_min50/plots/index.html      (reco-only, CSC/DT min cluster size 50)"
echo "  output_mds_min10/plots/index.html      (reco-only, CSC/DT min cluster size 10)"
echo "  output_mds_grid/plots/index.html      (gen-level: LLPs, efficiencies, sig-vs-bkg)"
