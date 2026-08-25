"""Study constants and the ``columns.yaml`` schema.

:data:`PARAM_SPEC` is the single source of truth for what a config set may
tune: one row per parameter, giving its default and its validator.  Adding a
parameter is one row -- the default, the type check and the error message all
come from it.  :func:`configure` turns a ``columns.yaml`` into a
:class:`Settings`, which is what :func:`columns.derive` and every helper below
it are handed.

Nothing is stored in module state.  A :class:`Settings` travels with the work,
so the processor can pickle it to a worker and two configurations (an eps scan
in a notebook) can be run side by side without one clobbering the other.

See ``docs/derived-columns.md`` for what the parameters mean physically.
"""

from __future__ import annotations

from typing import NamedTuple

# ── MDS study geometry ─────────────────────────────────────────────
LLP_PDGID = 999999

# Decay-volume definitions [cm] (standard MDS analysis geometry)
CSC_ZMIN, CSC_ZMAX, CSC_RMAX = 400.0, 1100.0, 695.5
DT_RMIN, DT_RMAX, DT_ZMAX = 380.0, 738.0, 650.0
RPC_EC_ZMIN, RPC_EC_ZMAX, RPC_EC_RMAX = 600.0, 1020.0, 660.0  # endcap; barrel = DT volume

RPC_MODES = ("separate", "merge", "match")


# ── Validators ─────────────────────────────────────────────────────
# Each takes the value and returns it (canonicalised), or raises ValueError
# with the reason; configure() prefixes the parameter name.

def positive(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0:
        raise ValueError(f"must be a positive number, got {v!r}")
    return v


def positive_int(v):
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"must be a positive integer, got {v!r}")
    return v


def boolean(v):
    if not isinstance(v, bool):
        raise ValueError(f"must be true or false, got {v!r}")
    return v


def one_of(*allowed):
    def check(v):
        if v not in allowed:
            raise ValueError(f"must be one of {allowed}, got {v!r}")
        return v
    return check


def text(v):
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"must be a non-empty string, got {v!r}")
    return v


def optional_str(v):
    if v is not None and not isinstance(v, str):
        raise ValueError(f"must be a string or null, got {v!r}")
    return v


def quantiles(v):
    if isinstance(v, str) or not hasattr(v, "__iter__") or not all(
            isinstance(q, (int, float)) and 0 < q <= 1 for q in v):
        raise ValueError("must be a list of containment fractions in (0, 1], "
                         f"got {v!r}")
    return tuple(v)


def iso_objects(v):
    """``{dR field: {collection, expression}}`` -- see docs/derived-columns.md."""
    if not isinstance(v, dict):
        raise ValueError("must be a mapping of dR field -> "
                         f"{{collection, expression}}, got {v!r}")
    for field, selection in v.items():
        if not str(field).isidentifier():
            raise ValueError(f"key '{field}' becomes a cluster field name, so it "
                             "must be an identifier")
        if not isinstance(selection, dict) or set(selection) != {"collection",
                                                                 "expression"}:
            raise ValueError(f"['{field}'] needs exactly 'collection' and "
                             f"'expression', got {selection!r}")
        for key in ("collection", "expression"):
            if not isinstance(selection[key], str) or not selection[key].strip():
                raise ValueError(f"['{field}'].{key} must be a non-empty string, "
                                 f"got {selection[key]!r}")
    return v


DEFAULT_ISO = {
    "drMuon": {
        "collection": "Muon",
        "expression": "(obj.pt > 10) & (abs(obj.eta) < 2.4) & obj.looseId",
    },
    "drJet": {
        "collection": "Jet",
        "expression": ("(obj.pt > 20) & (abs(obj.eta) < 2.4)"
                       " & (obj.neHEF < 0.8) & (obj.chHEF > 0.1)"
                       " & jet_id(obj, 'tightlepveto')"),
    },
}

# ── The schema: parameter -> (default, validator) ──────────────────
PARAM_SPEC = {
    "cluster_eps":         (0.4, positive),
    "cluster_min_samples": (10, positive_int),
    "rpc_min_samples":     (10, positive_int),
    "rpc_mode":            ("separate", one_of(*RPC_MODES)),
    "match_min_hits":      (10, positive_int),
    "oot_time_cut":        (12.5, positive),
    "pair_max_hits":       (2000, positive_int),
    "dr_quantiles":        ((0.5, 0.8, 0.9), quantiles),
    "llpidx_convention":   ("genpart", one_of("genpart", "ordinal")),
    "jerc":                (True, boolean),
    "jerc_era":            ("2024_Summer24", text),
    "jerc_algo":           ("AK4PFPuppi", text),
    "jerc_data_tag":       (None, optional_str),
    "iso_objects":         (DEFAULT_ISO, iso_objects),
}

DEFAULT_PARAMS = {name: default for name, (default, _) in PARAM_SPEC.items()}

# Optional helpers, in the order derive() runs them, each with the steps it
# needs.  A config's ``steps:`` list selects a subset; the default is all.
STEP_DEPS = {
    "jerc": (),
    "clusters": (),
    "cluster_isolation": ("clusters",),
    "llp": (),
    "llp_hits": ("llp",),
    "llp_reco": ("clusters", "llp_hits"),
    "llp_shape": ("llp",),
}


class Settings(NamedTuple):
    """One resolved ``columns.yaml``: what ``derive()`` runs on.

    Immutable and picklable, so the processor hands the same object to every
    worker instead of each of them re-reading a config that may not have
    reached them.
    """

    params: dict
    steps: tuple

    def __getitem__(self, key):
        """``settings["cluster_eps"]`` -- the helpers only ever read parameters."""
        if isinstance(key, str):
            try:
                return self.params[key]
            except KeyError:
                raise KeyError(f"no such parameter '{key}'; known: "
                               f"{sorted(PARAM_SPEC)}") from None
        return tuple.__getitem__(self, key)

    def replace(self, **overrides) -> "Settings":
        """A copy with some parameters changed (for tests and notebooks)."""
        return configure({"parameters": {**self.params, **overrides},
                          "steps": list(self.steps)})


def configure(cfg=None) -> Settings:
    """Validate a ``columns.yaml`` mapping -> :class:`Settings` (None = defaults).

    Recognised keys: ``parameters`` (see :data:`PARAM_SPEC`) and ``steps`` (see
    :data:`STEP_DEPS`).  Unknown keys and unsatisfied step dependencies raise,
    since a silently ignored typo here means silently wrong histograms.
    """
    cfg = cfg or {}
    unknown = set(cfg) - {"parameters", "steps"}
    if unknown:
        raise ValueError(f"columns.yaml: unknown key(s) {sorted(unknown)}; "
                         "expected 'parameters' and/or 'steps'")

    overrides = cfg.get("parameters") or {}
    unknown = set(overrides) - set(PARAM_SPEC)
    if unknown:
        raise ValueError(f"columns.yaml parameters: unknown {sorted(unknown)}; "
                         f"known: {sorted(PARAM_SPEC)}")

    params = {}
    for name, (default, validate) in PARAM_SPEC.items():
        value = overrides.get(name, default)
        try:
            params[name] = validate(value)
        except ValueError as e:
            raise ValueError(f"columns.yaml: {name} {e}") from None

    steps = list(cfg["steps"] or []) if "steps" in cfg else list(STEP_DEPS)
    unknown = [s for s in steps if s not in STEP_DEPS]
    if unknown:
        raise ValueError(f"columns.yaml steps: unknown {unknown}; "
                         f"known: {list(STEP_DEPS)}")
    for step in steps:
        missing = [d for d in STEP_DEPS[step] if d not in steps]
        if missing:
            raise ValueError(f"columns.yaml steps: '{step}' needs {missing}, "
                             "which this config does not enable")
    # Canonical order, so the list in the config is a set and not a schedule.
    return Settings(params, tuple(s for s in STEP_DEPS if s in steps))


DEFAULTS = configure()


def _dbscan_params(system, settings):
    """(eps, min_samples) for one rechit system."""
    key = "rpc_min_samples" if system == "rpc" else "cluster_min_samples"
    return settings["cluster_eps"], settings[key]
