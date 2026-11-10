"""Loading a config set: every file of it, and the custom columns it runs.

:func:`load_config_set` is the one place a set is read.  ``suep-run``,
``suep-config check`` and ``suep-config fields`` all start from the
:class:`ConfigSet` it returns, so they cannot disagree about what a set says.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import REPO_ROOT, load_config_file
from .corrections import load_correction_defs
from .histdefs import load_histogram_defs, load_selection_defs
from .inputs import load_samples
from .reweights import build_reweighters, load_reweight_defs


@dataclass
class ConfigSet:
    """Everything a run reads from one config set, loaded and resolved."""

    config_dir: Path
    samples: dict                 # name -> the sample's mapping
    histograms: dict              # name -> Histogram
    selections: dict              # name -> Selection
    corrections: dict             # corrections.yaml, as written
    reweighters: dict             # name -> Reweighter
    derive_fn: Callable | None    # custom/columns.py's derive()
    settings: object              # the resolved columns.yaml derive() runs on
    validate_fn: Callable | None  # custom/columns.py's validate()


def load_columns_config(path: str) -> dict:
    """Read a config's optional ``columns.yaml`` -> settings for ``derive()``.

    Missing file = ``{}`` = the module's own defaults.  The schema is the
    custom-columns module's business (``parameters`` / ``steps`` for the MDS
    module); it is only forwarded here.
    """
    return load_config_file(path)


def import_custom():
    """Import the ``custom.columns`` module that ships next to the package.

    Fatal when it is missing or does not import: every config set's derived
    collections (the clusters, the LLPs) come from it, so running without it
    would only produce empty histograms.
    """
    if not (REPO_ROOT / "custom" / "columns.py").exists():
        raise SystemExit(
            f"ERROR: custom/columns.py not found next to the suep_plot package "
            f"(looked in {REPO_ROOT}).\n  suep-plotting must be installed "
            "editable from its checkout:  pip install -e <checkout>")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module("custom.columns")
        # columns.py imports its helpers from sibling modules, so reload the
        # whole package: reloading columns.py alone would re-bind its names to
        # the already-cached submodules and an edit there would be ignored.
        # (This is for a live notebook kernel; a fresh suep-run imports once.)
        # In dependency order -- params.py owns the schema the other modules
        # import, so it goes first and columns.py last.
        helpers = sorted(n for n in sys.modules if n.startswith("custom.")
                         and n not in ("custom.columns", "custom.params"))
        for name in ["custom.params", *helpers]:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        importlib.reload(mod)
    except Exception as e:  # noqa: BLE001 - re-raised as a fatal, explained error
        raise SystemExit(
            f"ERROR: custom/columns.py failed to import: {type(e).__name__}: {e}\n"
            "  It needs scikit-learn, numba and awkward in the active "
            "environment (conda activate rootmds).") from e
    return mod


def load_custom_columns(columns_cfg: dict | None = None, verbose: bool = True):
    """Import custom/columns.py -> (derive_fn, settings, validate_fn).

    The settings are resolved here, once, and handed to the processor -- which
    passes them to ``derive()`` and pickles them to its workers, so a worker
    cannot end up running on settings its parent never chose.  A bad
    columns.yaml is fatal: it changes what the derived columns *mean*, so it
    must not degrade to a warning and silently different histograms.
    """
    mod = import_custom()
    derive_fn = getattr(mod, "derive", None)
    validate_fn = getattr(mod, "validate", None)
    configure = getattr(mod, "configure", None)
    if configure is None:
        if columns_cfg:
            raise SystemExit("ERROR: columns.yaml given, but custom/columns.py "
                             "has no configure() to read it")
        return derive_fn, None, validate_fn
    try:
        settings = configure(columns_cfg)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ERROR in columns.yaml: {e}")
    if derive_fn is not None and verbose:
        print("Custom columns: loaded derive()")
        print(f"                steps  {list(settings.steps)}")
        print(f"                params {settings.params}")
    return derive_fn, settings, validate_fn


def load_config_set(config_dir: str | Path, problems: list | None = None,
                    verbose: bool = False) -> ConfigSet:
    """Read one config set.

    A file that does not load is fatal (``SystemExit`` with the reason).  So
    are a bad ``reweights.yaml`` map and a bad ``columns.yaml`` -- unless a
    *problems* list is given: then those two are appended to it and the set
    is returned without reweighters / without custom columns, so a check can
    go on and report everything it finds in one pass.

    *verbose* prints the custom-column steps and parameters, as ``suep-run``
    does.
    """
    config_dir = Path(config_dir)
    samples = load_samples(config_dir / "samples.yaml")
    histograms = load_histogram_defs(config_dir / "histograms.yaml")
    selections = load_selection_defs(config_dir / "selections.yaml")
    corrections = load_correction_defs(config_dir / "corrections.yaml")
    reweight_defs = load_reweight_defs(config_dir / "reweights.yaml")

    # Reweighting changes physics results: a bad map is an error, never a
    # warning.
    try:
        reweighters = build_reweighters(reweight_defs, config_dir)
    except (ValueError, FileNotFoundError, KeyError) as e:
        if problems is None:
            raise SystemExit(f"ERROR in reweights.yaml: {e}")
        problems.append(f"reweights.yaml: {e}")
        reweighters = {}

    try:
        derive_fn, settings, validate_fn = load_custom_columns(
            load_columns_config(config_dir / "columns.yaml"), verbose=verbose)
    except SystemExit as e:
        if problems is None:
            raise
        problems.append(str(e))
        derive_fn = settings = validate_fn = None

    return ConfigSet(config_dir, samples, histograms, selections, corrections,
                     reweighters, derive_fn, settings, validate_fn)
