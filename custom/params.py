"""Study constants, tunable parameters and the ``columns.yaml`` reader.

Everything the other modules of this package read at run time lives here:
the LLP pdgId and decay-volume geometry, :data:`DEFAULT_PARAMS`,
:data:`STEP_DEPS`, and the live settings :data:`PARAMS` / :data:`STEPS` that
:func:`configure` fills from a config directory's ``columns.yaml``.

:data:`PARAMS` and :data:`STEPS` are mutated in place, so ``from .params import
PARAMS`` in another module keeps seeing the current settings.
"""

from __future__ import annotations

# ── MDS study parameters ───────────────────────────────────────────
LLP_PDGID = 999999

# Decay-volume definitions [cm] (standard MDS analysis geometry)
CSC_ZMIN, CSC_ZMAX, CSC_RMAX = 400.0, 1100.0, 695.5
DT_RMIN, DT_RMAX, DT_ZMAX = 380.0, 738.0, 650.0
RPC_EC_ZMIN, RPC_EC_ZMAX, RPC_EC_RMAX = 600.0, 1020.0, 660.0  # endcap; barrel = DT volume

# ── Tunable parameters ─────────────────────────────────────────────
# Set per config set in columns.yaml (``parameters:``); see configure().
#
# ``cluster_eps`` / ``cluster_min_samples``: DBSCAN eps (dR radius in eta-phi)
# and minimum cluster size for CSC and DT (50 = standard MDS analysis).  RPC
# keeps its own, smaller ``rpc_min_samples`` (sparse system).
# ``match_min_hits``: cluster <-> LLP truth matching, matched := (# rechits
# sharing one llpIdx) >= this.
# ``pair_max_hits``: an LLP with more matched rechits than this is strided down
# before the O(N^2) pairwise dR matrix is built (hit multiplicities are far
# below the cap in practice, so this never fires).
# ``dr_quantiles``: containment fractions for the per-LLP cone radius (field
# names dr50/dr80/...).
# ``llpidx_convention``: how the rechit ``llpIdx`` branch is read — "genpart"
# (llpIdx is the SUEPGenPart index, post-Geant4-fix files) or "ordinal"
# (llpIdx is the ordinal LLP index within the event, pre-fix files).
#
# Cluster isolation (``cluster_isolation`` step): which reconstructed muons and
# jets count as prompt activity for ``drMuon`` / ``drJet``.  Note JEC/JER
# rescale pT and mass but leave eta/phi untouched, so corrections reach dR only
# through which jets pass ``iso_jet_pt``.
# ``jerc``: apply L1L2L3Res JEC to events.Jet before the jets are selected (the
# ``jerc`` step), plus JER smearing on MC.  Data is corrected with the era's
# ``*_DATA`` tag — L1L2L3Res is where the residual corrections live — and is
# never smeared.  ``jerc_data_tag`` overrides that tag; None takes the era entry
# from suep_plot.jme.DEFAULTS.
# ``iso_objects``: which prompt objects the cluster dR is measured against, one
# entry per dR field.  Each entry is a ``collection`` (a NanoAOD collection name)
# and an ``expression`` returning a per-object boolean.  Nothing about muons or
# jets is baked into the code, so a new field -- electrons, photons, taus, HLT
# jets -- is a config edit, not a code edit::
#
#     iso_objects:
#       drMuon:      {collection: Muon, expression: "obj.pt > 10 & obj.looseId"}
#       drElectron:  {collection: Electron, expression: "obj.pt > 15"}
#
# In scope: ``obj`` (the collection), ``events``/``ev``, ``ak``, ``np``, the safe
# builtins, and ``jet_id(obj, "tight"|"tightlepveto")`` -- the official jsonpog
# ``jetid.json.gz`` decision, since 2024 NanoAOD no longer stores
# ``Jet_jetId``.
DEFAULT_PARAMS = {
    "cluster_eps": 0.4,
    "cluster_min_samples": 10,
    "rpc_min_samples": 10,
    "match_min_hits": 10,
    "pair_max_hits": 2000,
    "dr_quantiles": (0.5, 0.8, 0.9),
    "llpidx_convention": "genpart",
    "jerc": True,
    "jerc_era": "2024_Summer24",
    "jerc_algo": "AK4PFPuppi",
    "jerc_data_tag": None,
    "iso_objects": {
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
    },
}

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

# Live settings, filled by configure() below; mutated in place, never rebound.
PARAMS = {}
STEPS = []


def configure(cfg=None):
    """Apply a config directory's ``columns.yaml`` (``{}``/None = defaults).

    Recognized keys: ``parameters`` (see :data:`DEFAULT_PARAMS`) and ``steps``
    (see :data:`STEP_DEPS`).  Called by the processor once per run, before the
    first chunk; unknown keys and unsatisfied step dependencies raise, since a
    silently ignored typo here would mean silently wrong histograms.
    """
    cfg = cfg or {}
    unknown = set(cfg) - {"parameters", "steps"}
    if unknown:
        raise ValueError(f"columns.yaml: unknown key(s) {sorted(unknown)}; "
                         "expected 'parameters' and/or 'steps'")

    params = cfg.get("parameters") or {}
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"columns.yaml parameters: unknown {sorted(unknown)}; "
                         f"known: {sorted(DEFAULT_PARAMS)}")

    # Mutated in place, never rebound: the other modules of this package
    # hold a reference to these very objects.
    PARAMS.clear()
    PARAMS.update(DEFAULT_PARAMS)
    PARAMS.update(params)
    if PARAMS["llpidx_convention"] not in ("genpart", "ordinal"):
        raise ValueError("columns.yaml: llpidx_convention must be 'genpart' or "
                         f"'ordinal', got '{PARAMS['llpidx_convention']}'")
    for key in ("cluster_min_samples", "rpc_min_samples", "match_min_hits",
                "pair_max_hits"):
        if not isinstance(PARAMS[key], int) or PARAMS[key] < 1:
            raise ValueError(f"columns.yaml: {key} must be a positive integer, "
                             f"got {PARAMS[key]!r}")
    iso = PARAMS["iso_objects"]
    if not isinstance(iso, dict):
        raise ValueError("columns.yaml: iso_objects must be a mapping of "
                         f"dR field -> {{collection, expression}}, got {iso!r}")
    for field, selection in iso.items():
        if not isinstance(selection, dict) or set(selection) != {"collection",
                                                                 "expression"}:
            raise ValueError(
                f"columns.yaml: iso_objects['{field}'] needs exactly "
                f"'collection' and 'expression', got {selection!r}")
        for key in ("collection", "expression"):
            if not isinstance(selection[key], str) or not selection[key].strip():
                raise ValueError(f"columns.yaml: iso_objects['{field}'].{key} "
                                 f"must be a non-empty string, got "
                                 f"{selection[key]!r}")
        if not str(field).isidentifier():
            raise ValueError(f"columns.yaml: iso_objects key '{field}' becomes a "
                             "cluster field name, so it must be an identifier")
    if not PARAMS["cluster_eps"] > 0:
        raise ValueError("columns.yaml: cluster_eps must be positive, got "
                         f"{PARAMS['cluster_eps']!r}")
    quantiles = PARAMS["dr_quantiles"]
    if (isinstance(quantiles, str) or not hasattr(quantiles, "__iter__")
            or not all(isinstance(q, (int, float)) and 0 < q <= 1 for q in quantiles)):
        raise ValueError("columns.yaml: dr_quantiles must be a list of "
                         f"containment fractions in (0, 1], got {quantiles!r}")
    PARAMS["dr_quantiles"] = tuple(quantiles)

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
    STEPS[:] = [s for s in STEP_DEPS if s in steps]
    return PARAMS, tuple(STEPS)


def _dbscan_params(system):
    """(eps, min_samples) for one rechit system."""
    key = "rpc_min_samples" if system == "rpc" else "cluster_min_samples"
    return PARAMS["cluster_eps"], PARAMS[key]


# Defaults in place for a bare ``import custom.columns`` (notebooks, scripts)
# that never calls configure().
configure()
