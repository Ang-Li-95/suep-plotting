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
SETS = [p for p in ALL if not read_set_file(p).get("abstract")]


def test_the_sets_were_found():
    assert len(SETS) >= 15, "configs/ layout moved? found: " + ", ".join(
        p.name for p in SETS)


def _parents(config_dir):
    """Every set a set's files or set.yaml extend."""
    import yaml
    out = set()
    ext = read_set_file(config_dir).get("extends")
    if ext:
        out.add(ext)
    for f in config_dir.glob("*.yaml"):
        raw = yaml.safe_load(f.read_text()) or {}
        ext = raw.get("_extends") if isinstance(raw, dict) else None
        out.update([ext] if isinstance(ext, str) else ext or [])
    return {(config_dir / e).resolve() for e in out}


def test_no_set_relies_on_a_shared_base():
    """The shared bases are gone (2026-11-17): a set is self-contained or a
    variant of one runnable study, never an extension of an abstract base."""
    assert [p.name for p in ALL if read_set_file(p).get("abstract")] == []
    for config_dir in SETS:
        for parent in _parents(config_dir):
            assert not read_set_file(parent).get("abstract"), \
                f"{config_dir.name} extends the base {parent.name}"


@pytest.mark.parametrize("config_dir", SETS, ids=lambda p: p.name)
def test_config_set_is_valid(config_dir):
    assert validate_config_set(config_dir) == []
