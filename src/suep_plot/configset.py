"""Loading a config set: every file of it, and the custom columns it runs.

:func:`load_config_set` is what ``suep-config`` reads a set with;
:func:`load_custom_columns` imports ``custom/columns.py`` and resolves the
set's ``columns.yaml`` into the settings ``derive()`` runs on.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .config import load_config_file
from .corrections import load_correction_defs
from .histdefs import load_histogram_defs, load_selection_defs
from .inputs import load_samples
from .reweights import build_reweighters, load_reweight_defs


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
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / "custom" / "columns.py").exists():
        raise SystemExit(
            f"ERROR: custom/columns.py not found next to the suep_plot package "
            f"(looked in {repo_root}).\n  suep-plotting must be installed "
            "editable from its checkout:  pip install -e <checkout>")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
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
    """Import custom/columns.py -> (derive_fn, settings).

    The settings are resolved here, once, and handed to the processor -- which
    passes them to ``derive()`` and pickles them to its workers, so a worker
    cannot end up running on settings its parent never chose.  A bad
    columns.yaml is fatal: it changes what the derived columns *mean*, so it
    must not degrade to a warning and silently different histograms.
    """
    mod = import_custom()
    derive_fn = getattr(mod, "derive", None)
    configure = getattr(mod, "configure", None)
    if configure is None:
        if columns_cfg:
            raise SystemExit("ERROR: columns.yaml given, but custom/columns.py "
                             "has no configure() to read it")
        return derive_fn, None
    try:
        settings = configure(columns_cfg)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ERROR in columns.yaml: {e}")
    if derive_fn is not None and verbose:
        print("Custom columns: loaded derive()")
        print(f"                steps  {list(settings.steps)}")
        print(f"                params {settings.params}")
    return derive_fn, settings


def load_config_set(config_dir: str | Path) -> dict:
    """Everything a run reads from one config set, loaded and resolved.

    Keys: ``samples``, ``histograms``, ``selections``, ``corrections``,
    ``reweighters``, ``derive_fn``, ``settings``.
    """
    config_dir = Path(config_dir)
    try:
        reweighters = build_reweighters(
            load_reweight_defs(config_dir / "reweights.yaml"), config_dir)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"ERROR in reweights.yaml: {e}")
    derive_fn, settings = load_custom_columns(
        load_columns_config(config_dir / "columns.yaml"), verbose=False)
    return {
        "samples": load_samples(config_dir / "samples.yaml"),
        "histograms": load_histogram_defs(config_dir / "histograms.yaml"),
        "selections": load_selection_defs(config_dir / "selections.yaml"),
        "corrections": load_correction_defs(config_dir / "corrections.yaml"),
        "reweighters": reweighters,
        "derive_fn": derive_fn,
        "settings": settings,
    }
