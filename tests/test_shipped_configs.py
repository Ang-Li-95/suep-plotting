"""Every config set shipped in configs/ must load and pass the static checks.

This is what would have caught configs_mds_trigger running the
cluster_isolation step without the selections it reads: nothing opened that
set between the change that broke it and the review that found it.
"""

from pathlib import Path

import pytest

from suep_plot.processor import validate_config_set

REPO = Path(__file__).resolve().parents[1]

# Bases are only ever inherited from; on their own they are incomplete on
# purpose (common_clustersrc leaves the population cut to its children).
BASES = {"common", "common_gen", "common_clustersrc"}

SETS = sorted(p for p in (REPO / "configs").iterdir()
              if (p / "histograms.yaml").exists() and p.name not in BASES)


def test_the_sets_were_found():
    assert len(SETS) >= 15, "configs/ layout moved? found: " + ", ".join(
        p.name for p in SETS)


@pytest.mark.parametrize("config_dir", SETS, ids=lambda p: p.name)
def test_config_set_is_valid(config_dir):
    assert validate_config_set(config_dir) == []
