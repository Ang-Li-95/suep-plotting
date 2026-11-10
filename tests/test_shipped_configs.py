"""Every config set shipped in configs/ must load and pass the static checks.

This is what would have caught configs_mds_trigger running the
cluster_isolation step without the selections it reads: nothing opened that
set between the change that broke it and the review that found it.
"""

from pathlib import Path

import pytest

from suep_plot.cli import _is_config_set
from suep_plot.config import read_set_file
from suep_plot.checks import validate_config_set

REPO = Path(__file__).resolve().parents[1]

ALL = sorted(p for p in (REPO / "configs").iterdir() if _is_config_set(str(p)))
# Bases (abstract: true in set.yaml) are only ever inherited from; on their
# own they are incomplete on purpose (common_clustersrc leaves the population
# cut to its children).
SETS = [p for p in ALL if not read_set_file(p).get("abstract")]


def test_the_sets_were_found():
    assert len(SETS) >= 15, "configs/ layout moved? found: " + ", ".join(
        p.name for p in SETS)
    assert {p.name for p in ALL} - {p.name for p in SETS} == \
        {"common", "common_gen", "common_clustersrc"}


@pytest.mark.parametrize("config_dir", SETS, ids=lambda p: p.name)
def test_config_set_is_valid(config_dir):
    assert validate_config_set(config_dir) == []
