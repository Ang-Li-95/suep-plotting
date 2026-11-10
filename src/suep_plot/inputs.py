"""A config set's samples and their input files.

``samples.yaml`` to sample definitions, a sample's ``files:`` (files, globs,
directories, local or xrootd) to a flat file list, and one file opened as
NanoEvents.
"""

from __future__ import annotations

import difflib
from pathlib import Path

# Applies the scipy shim before coffea.lookup_tools is imported anywhere.
from . import _compat  # noqa: F401

from coffea.nanoevents import NanoAODSchema

from .config import load_config_file

# MDSNano is a custom NanoAOD flavour; suppress cross-reference warnings for
# collections without the standard NanoAOD index branches.
NanoAODSchema.warn_missing_crossrefs = False


# ── samples ───────────────────────────────────────────────────────


def load_samples(path: str) -> dict:
    """Read a config's ``samples.yaml`` -> {name: cfg}.

    Definitions normally come from the shared registry named by ``_registry:``;
    see :mod:`suep_plot.config` for that and the other config directives.
    """
    return load_config_file(path)


def select_samples(sample_defs: dict, wanted: list[str] | None) -> dict:
    """The samples to process: all of them, or the ones named in *wanted*.

    Names may also be given comma-separated (``-s a,b``).  An unknown name is
    fatal -- silently processing fewer samples than asked for looks like a
    successful run -- and the message lists what the set does define.
    """
    if not sample_defs:
        raise SystemExit("ERROR: samples.yaml defines no samples")
    if not wanted:
        return dict(sample_defs)
    names = [n.strip() for spec in wanted for n in str(spec).split(",") if n.strip()]
    unknown = [n for n in names if n not in sample_defs]
    if unknown:
        hints = []
        for n in unknown:
            close = difflib.get_close_matches(n, list(sample_defs), n=1)
            if close:
                hints.append(f"'{n}' -> did you mean '{close[0]}'?")
        hint = ("\n  " + "\n  ".join(hints)) if hints else ""
        raise SystemExit(f"ERROR: unknown sample(s) {unknown}{hint}\n"
                         f"  available: {', '.join(sample_defs)}")
    return {k: v for k, v in sample_defs.items() if k in names}


# ── input files ───────────────────────────────────────────────────


def _split_xrootd_url(url: str) -> tuple[str, str]:
    """``root://host//eos/dir`` -> (``root://host/``, ``/eos/dir``)."""
    scheme, rest = url.split("://", 1)
    host, _, path = rest.partition("/")
    return f"{scheme}://{host}/", "/" + path.lstrip("/")


def _xrootd_listdir(fs, path: str, recursive: bool = True) -> list[str]:
    """List ``*.root`` under a remote directory (server paths).

    Subdirectories are walked too unless *recursive* is false.
    """
    from XRootD.client.flags import DirListFlags

    status, listing = fs.dirlist(path, DirListFlags.STAT)
    if not status.ok or listing is None:
        raise RuntimeError(f"xrootd dirlist failed for {path}: {status.message}")

    out = []
    for entry in listing:
        child = f"{path.rstrip('/')}/{entry.name}"
        if entry.statinfo is not None and entry.statinfo.flags & 2:  # kXR_isDir
            if recursive:
                out.extend(_xrootd_listdir(fs, child))
        elif entry.name.endswith(".root"):
            out.append(child)
    return sorted(out)


def _resolve_xrootd(spec: str) -> list[str]:
    """Expand one xrootd spec: a file passes through, a directory is walked,
    a trailing wildcard is matched against its parent directory's listing."""
    import fnmatch

    from XRootD import client

    prefix, path = _split_xrootd_url(spec)
    if "*" not in path and "?" not in path and path.endswith(".root"):
        return [spec]

    fs = client.FileSystem(prefix)
    # Server paths are absolute, so host + path keeps the usual root://host//eos/…
    if "*" in path or "?" in path:
        # Like a local glob: the pattern matches files in that directory only,
        # not same-named files further down.
        parent, _, pattern = path.rpartition("/")
        return [prefix + f for f in _xrootd_listdir(fs, parent, recursive=False)
                if fnmatch.fnmatch(f.rpartition("/")[2], pattern)]
    return [prefix + f for f in _xrootd_listdir(fs, path)]


def _resolve_files(file_specs: list[str]) -> list[str]:
    """Expand globs / directories / xrootd URLs -> flat list of file paths.

    A spec may be a single file, a glob, or a directory (local or xrootd), in
    which case it is walked recursively for ``*.root``.
    """
    import glob
    import os

    out = []
    for spec in file_specs:
        if "://" in spec:
            out.extend(_resolve_xrootd(spec))
        elif "*" in spec or "?" in spec:
            out.extend(sorted(glob.glob(spec)))
        elif os.path.isdir(spec):
            out.extend(sorted(str(p) for p in Path(spec).rglob("*.root")))
        else:
            out.append(spec)
    return out


def resolve_sample_files(cfg: dict) -> list[str]:
    """The files of one sample, honouring an optional ``max_files:`` cap.

    ``max_files`` takes the first N of the resolved list.  The list is sorted,
    so a capped run is reproducible and can be widened later.
    """
    files = _resolve_files(cfg.get("files") or [])
    cap = cfg.get("max_files")
    return files[:int(cap)] if cap else files


def open_events(path: str, tree: str = "Events", n: int = 200):
    """The first *n* events of one file as NanoEvents, read lazily."""
    from coffea.nanoevents import NanoEventsFactory

    # "virtual" mode reads branches lazily, so a check only touches the
    # branches the expressions actually use (an eager open materializes every
    # branch -- seconds of wasted I/O on files with PFCand etc.).
    try:
        return NanoEventsFactory.from_root(
            {path: tree}, mode="virtual", schemaclass=NanoAODSchema,
            entry_stop=n).events()
    except TypeError:
        return NanoEventsFactory.from_root(
            {path: tree}, mode="eager", schemaclass=NanoAODSchema,
            entry_stop=n).events()
