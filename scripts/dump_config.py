#!/usr/bin/env python3
"""Dump a config set's fully expanded definitions as canonical YAML.

Every loader in the package is run over the set and its result printed sorted,
so two revisions of the config machinery can be compared with ``diff``:

    python scripts/dump_config.py configs/configs_mds > before.yaml

This is the acceptance check for refactors of the config layer -- an empty diff
means the expanded histograms, selections, samples and settings are unchanged,
without having to reprocess any ROOT file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _plain(obj):
    """Strip tuples/sets/Path so two runs serialize identically."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_plain(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _resolved_columns(config_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    from custom import columns

    from suep_plot.processor import load_columns_config

    settings = columns.configure(load_columns_config(config_dir / "columns.yaml"))
    return {"parameters": dict(settings.params), "steps": list(settings.steps)}


def dump(config_dir: Path) -> dict:
    from suep_plot.corrections import load_correction_defs
    from suep_plot.histograms import load_histogram_defs, load_selection_defs
    from suep_plot.plot import load_derived_plot_defs
    from suep_plot.processor import load_columns_config, load_samples
    from suep_plot.reweights import load_reweight_defs

    loaders = {
        "samples": lambda p: load_samples(p / "samples.yaml"),
        "histograms": lambda p: load_histogram_defs(p / "histograms.yaml"),
        "selections": lambda p: load_selection_defs(p / "selections.yaml"),
        "corrections": lambda p: load_correction_defs(p / "corrections.yaml"),
        "reweights": lambda p: load_reweight_defs(p / "reweights.yaml"),
        "derived_plots": lambda p: load_derived_plot_defs(str(p / "derived_plots.yaml")),
        "columns": lambda p: load_columns_config(p / "columns.yaml"),
        # The settings derive() actually runs on, not the YAML that produced
        # them: two configs that spell the defaults out differently are the
        # same run, and this is the level a refactor has to preserve.
        "columns_resolved": _resolved_columns,
    }
    out = {}
    for section, load in loaders.items():
        try:
            out[section] = _plain(load(config_dir) or {})
        except (Exception, SystemExit) as e:  # recorded, so a regression shows up
            out[section] = {"__error__": f"{type(e).__name__}: {e}"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_dir", type=Path, nargs="+")
    ap.add_argument("-o", "--out-dir", type=Path, default=None,
                    help="Write <set>.yaml per config set instead of printing "
                         "(one process for all of them: the imports dominate)")
    args = ap.parse_args(argv)
    for cdir in args.config_dir:
        text = yaml.safe_dump(dump(cdir), sort_keys=True,
                              default_flow_style=False, width=100)
        if args.out_dir is None:
            print(text)
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{cdir.name}.yaml").write_text(text)
            print(f"  {cdir.name}")


if __name__ == "__main__":
    main()
