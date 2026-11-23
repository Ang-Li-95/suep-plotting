"""ZeroBias overlay of the muon-system rechits on simulated events.

The 2024 background MC carries simulated pileup and noise in the muon
chambers, which the DY noise study found under-modelled.  A set with an
``overlay.yaml`` therefore rebuilds the CSC, DT and RPC rechits of every
simulated chunk before anything is derived from them:

1. **clean** -- keep only the simulated rechits that look like the hard
   process: in time, and within dR of a gen jet or of a reachable gen muon
   (its phi bent to the hit, :mod:`suep_plot.propagation`).  The working point
   is the noise study's (``plot_matching_performance.py``).
2. **pick** a certified ZeroBias event for every simulated one, never twice
   within a sample (:mod:`suep_plot.overlay_pool`).
3. optionally **clean the ZeroBias event** (DataClean): drop its rechits near
   one of its own prompt jets or muons (``jet_sel_foriso`` /
   ``muon_sel_foriso``).
4. **merge**: the kept simulated rechits followed by the ZeroBias ones.

Every other collection stays the simulated one.  Per system the inputs stay
available for validation -- ``<collection>MC`` with ``cleanKeep``,
``<collection>ZB`` with ``dataCleanKeep`` -- and ``<collection>`` itself
becomes the merged rechits, each flagged ``fromZeroBias``.
``events.overlay`` names the ZeroBias partner (``run``, ``luminosityBlock``,
``event``) and whether it is ``reused``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import awkward as ak

from .config import ConfigError, load_config_file
from .fill import compile_expr
from .overlay_pool import (PoolPlan, ZeroBiasPicker, build_sample_plan, plan_dir,
                           pool_definition)
from .propagation import min_dr, min_dr_bent, reachable_gen_muons

# The rechit collections the overlay rebuilds, and the timing field each one
# is cleaned on (DT rechits carry no time in MDSNano).
RECHIT_COLLECTIONS = ("cscRechits", "dtRecHits", "rpcRecHits")
CSC_TIME, RPC_BX = "Tpeak", "Bx"
# The prompt objects DataClean removes ZeroBias rechits around: the cluster
# isolation's selections (custom/helpers.py ISO_SELECTIONS), so "near a jet"
# means the same thing for a rechit as for a cluster.
DATA_CLEAN_SELECTIONS = ("jet_sel_foriso", "muon_sel_foriso")


@dataclass(frozen=True)
class CleaningCuts:
    """What a simulated rechit must pass to be kept (``mc_clean:``)."""

    csc_tpeak: tuple[float, float]   # ns, open window: lo < Tpeak < hi
    rpc_bx: tuple[int, ...]          # accepted bunch crossings
    dr: float                        # max dR to a gen jet or reachable gen muon
    genjet_pt: float                 # GeV: gen jets above it count


@dataclass(frozen=True)
class ZeroBiasSource:
    """Where the overlaid events come from (``zerobias:``)."""

    pool: str                        # datasets.yaml entry
    lumi_mask: str                   # certification JSON
    data_clean_dr: float | None      # None: DataClean off


@dataclass(frozen=True)
class OverlayConfig:
    """A set's ``overlay.yaml``, validated."""

    mc_clean: CleaningCuts
    zerobias: ZeroBiasSource | None  # None: clean only, add nothing


# ── on events ─────────────────────────────────────────────────────


class Overlay:
    """``overlay.yaml`` in action: what the processor runs on a simulated chunk.

    *picker* supplies the ZeroBias partners (needed when the config adds
    ZeroBias); *selections*, *derive_fn* and *settings* are the set's, which
    DataClean selects the ZeroBias prompt objects with.
    """

    def __init__(self, config: OverlayConfig, picker: ZeroBiasPicker | None = None,
                 selections=None, derive_fn=None, settings=None):
        if config.zerobias is not None and picker is None:
            raise ValueError("overlay: adding ZeroBias needs a ZeroBiasPicker")
        self.config = config
        self.picker = picker
        self.selections = selections or {}
        self.derive_fn = derive_fn
        self.settings = settings

    def apply(self, events, sample: str, mc_file: str, entry_start: int, entry_stop: int):
        """Rebuild the rechits of MC entries [entry_start, entry_stop) of *mc_file*."""
        events = clean_mc(events, self.config.mc_clean)
        source = self.config.zerobias
        if source is None:
            return events

        partners, reused = self.picker.pick(sample, mc_file, entry_start, entry_stop)
        if len(partners) != len(events):
            raise ValueError(f"overlay: {len(partners)} ZeroBias events for "
                             f"{len(events)} simulated ones ({mc_file})")
        if source.data_clean_dr is not None:
            partners = clean_zerobias(partners, source.data_clean_dr, self.selections,
                                      self.derive_fn, self.settings)
        events = add_zerobias(events, partners)
        events["overlay"] = ak.zip({
            "run": partners.run, "luminosityBlock": partners.luminosityBlock,
            "event": partners.event, "reused": reused})
        return events


def zerobias_collections(source: ZeroBiasSource) -> tuple[str, ...]:
    """What has to be read from a ZeroBias event: its rechits, and its prompt
    objects when DataClean is on."""
    if source.data_clean_dr is None:
        return RECHIT_COLLECTIONS
    return (*RECHIT_COLLECTIONS, "Jet", "Muon")


def overlay_for_run(cs, run_dir) -> Overlay | None:
    """The overlay of a run of config set *cs*, its pairs read from *run_dir*."""
    if cs.overlay is None:
        return None
    picker = None
    if cs.overlay.zerobias is not None:
        picker = ZeroBiasPicker(plan_dir(run_dir), zerobias_collections(cs.overlay.zerobias))
    return Overlay(cs.overlay, picker, cs.selections, cs.derive_fn, cs.settings)


def overlay_preview(cs, first_files: dict[str, str]) -> Overlay | None:
    """An overlay for checking expressions, with no run directory.

    Each sample's first file (*first_files*) gets one block: the files of the
    pool's first directory, read in their listed order.  Nothing is written.
    """
    if cs.overlay is None:
        return None
    source = cs.overlay.zerobias
    if source is None:
        return Overlay(cs.overlay)
    from .inputs import _resolve_files, file_specs

    first_spec = file_specs(pool_definition(source.pool))[:1]
    pool = PoolPlan(source.pool, source.lumi_mask, "", 0, tuple(_resolve_files(first_spec)))
    plans = {sample: build_sample_plan(sample, [mc_file], pool)
             for sample, mc_file in first_files.items()}
    picker = ZeroBiasPicker.in_memory(pool, plans, zerobias_collections(source))
    return Overlay(cs.overlay, picker, cs.selections, cs.derive_fn, cs.settings)


# ── 1. cleaning the simulated rechits ─────────────────────────────


def clean_mc(events, cuts: CleaningCuts):
    """Keep the simulated rechits that look like the hard process.

    A rechit is kept when it is in time (CSC ``Tpeak`` window, RPC ``Bx``; DT
    has no time) and within ``cuts.dr`` of a gen jet above ``cuts.genjet_pt``
    (plain dR to the jet axis) or of a reachable gen muon (dR to its phi bent
    to the hit).  Each input collection stays as ``<collection>MC`` with the
    per-hit decision ``cleanKeep``.
    """
    missing = [c for c in (*RECHIT_COLLECTIONS, "GenJet") if c not in events.fields]
    if missing:
        raise ValueError(f"overlay: the MC cleaning needs {missing}, which these "
                         "events do not have")
    genjets = events.GenJet[events.GenJet.pt > cuts.genjet_pt]
    gen_muons = reachable_gen_muons(_gen_particles(events))

    for collection in RECHIT_COLLECTIONS:
        hits = events[collection]
        near_gen = ((min_dr_bent(hits, gen_muons) < cuts.dr)
                    | (min_dr(hits, genjets) < cuts.dr))
        keep = _in_time(collection, hits, cuts) & near_gen
        kept = hits[keep]
        events[f"{collection}MC"] = ak.with_field(hits, keep, "cleanKeep")
        events[collection] = ak.with_field(
            kept, ak.zeros_like(kept.Eta, dtype=bool), "fromZeroBias")
    return events


def _gen_particles(events):
    """The full gen table where the file has it; the pruned GenPart holds
    exactly the same status-1 muons (DY noise study, punchthrough.py)."""
    return events.SUEPGenPart if "SUEPGenPart" in events.fields else events.GenPart


def _in_time(collection: str, hits, cuts: CleaningCuts):
    """Per-hit timing decision; every DT hit passes (no time in MDSNano)."""
    if collection == "cscRechits":
        lo, hi = cuts.csc_tpeak
        return (hits[CSC_TIME] > lo) & (hits[CSC_TIME] < hi)
    if collection == "rpcRecHits":
        in_window = hits[RPC_BX] == cuts.rpc_bx[0]
        for bx in cuts.rpc_bx[1:]:
            in_window = in_window | (hits[RPC_BX] == bx)
        return in_window
    return ak.ones_like(hits.Eta, dtype=bool)


# ── 3. DataClean ──────────────────────────────────────────────────


def clean_zerobias(partners, dr: float, selections: dict, derive_fn, settings):
    """Mark the ZeroBias rechits away from the event's own prompt objects.

    Each rechit collection gets ``dataCleanKeep``: False within *dr* of a
    ``jet_sel_foriso`` jet (plain dR to the axis) or a ``muon_sel_foriso`` muon
    (its phi bent to the hit, as for the gen muons).  The jet ID those
    selections read comes from the set's own ``jet_id`` step; the jets keep
    their stored NanoAOD pT (no JEC is rerun on ZeroBias).
    """
    if derive_fn is None or settings is None:
        raise ValueError("overlay: DataClean needs the custom columns (custom/"
                         "columns.py) for the jet ID")
    partners = derive_fn(partners, settings._replace(steps=("jet_id",)), selections)
    jets = _selected_objects(partners, selections, "jet_sel_foriso")
    muons = _selected_objects(partners, selections, "muon_sel_foriso")
    muon_tracks = ak.zip({"eta": muons.eta, "phi": muons.phi, "pt": muons.pt,
                          "charge": muons.charge})

    for collection in RECHIT_COLLECTIONS:
        hits = partners[collection]
        near_prompt = ((min_dr(hits, jets) < dr)
                       | (min_dr_bent(hits, muon_tracks) < dr))
        partners[collection] = ak.with_field(hits, ~near_prompt, "dataCleanKeep")
    return partners


def _selected_objects(events, selections: dict, name: str):
    selection = selections.get(name)
    if selection is None or not selection.collection:
        raise ValueError(f"overlay: DataClean needs the object selection '{name}' "
                         "(with a 'collection:') in selections.yaml")
    try:
        keep = compile_expr(selection.expression)(events)
    except Exception as e:
        raise ValueError(f"overlay: '{name}' failed on the ZeroBias events: "
                         f"{type(e).__name__}: {e}") from e
    return events[selection.collection][keep]


# ── 4. merging ────────────────────────────────────────────────────


def add_zerobias(events, partners):
    """Merge the kept simulated rechits with the kept ZeroBias ones.

    ``<collection>MC`` (from :func:`clean_mc`) and the partners' collection must
    carry the same rechit fields: a field only one side has would be missing
    or undefined for half of the merged hits.
    """
    for collection in RECHIT_COLLECTIONS:
        mc_hits = events[f"{collection}MC"]
        zb_hits = partners[collection]
        if "dataCleanKeep" not in ak.fields(zb_hits):
            zb_hits = ak.with_field(zb_hits, ak.ones_like(zb_hits.Eta, dtype=bool),
                                    "dataCleanKeep")
        fields = [f for f in ak.fields(mc_hits) if f != "cleanKeep"]
        zb_fields = [f for f in ak.fields(zb_hits) if f != "dataCleanKeep"]
        if set(fields) != set(zb_fields):
            raise ValueError(
                f"overlay: {collection} has different fields in the simulated and "
                f"the ZeroBias events: only simulated {sorted(set(fields) - set(zb_fields))}, "
                f"only ZeroBias {sorted(set(zb_fields) - set(fields))}")

        events[f"{collection}ZB"] = zb_hits
        merged = ak.concatenate(
            [_hits_with_origin(mc_hits[mc_hits.cleanKeep], fields, from_zerobias=False),
             _hits_with_origin(zb_hits[zb_hits.dataCleanKeep], fields, from_zerobias=True)],
            axis=1)
        events[collection] = merged
    return events


def _hits_with_origin(hits, fields, from_zerobias: bool):
    """Plain records of *fields* in one order, plus ``fromZeroBias``, so the
    simulated and ZeroBias halves concatenate into one record type."""
    return ak.zip({**{f: hits[f] for f in fields},
                   "fromZeroBias": ak.full_like(hits.Eta, from_zerobias, dtype=bool)})


# ── overlay.yaml ──────────────────────────────────────────────────


_TOP_KEYS = ("mc_clean", "add_zerobias", "zerobias")
_REQUIRED_TOP_KEYS = ("mc_clean", "add_zerobias")
_CLEAN_KEYS = ("csc_tpeak", "rpc_bx", "dr", "genjet_pt")
_ZEROBIAS_KEYS = ("pool", "lumi_mask", "data_clean")


def load_overlay_config(path: str | Path) -> OverlayConfig | None:
    """Read a set's ``overlay.yaml``; None when the set has none.

    Every key is required and an unknown key is an error: a cut that silently
    took a default would clean differently from what the file says.
    """
    raw = load_config_file(path)
    if not raw:
        return None
    where = f"ERROR in {path}"
    _require_keys(raw, _TOP_KEYS, _REQUIRED_TOP_KEYS, where)

    add_zerobias = raw["add_zerobias"]
    if not isinstance(add_zerobias, bool):
        raise ConfigError(f"{where}: add_zerobias must be true or false, "
                          f"got {add_zerobias!r}")
    if add_zerobias and "zerobias" not in raw:
        raise ConfigError(f"{where}: add_zerobias is true but there is no 'zerobias:'")
    if not add_zerobias and "zerobias" in raw:
        raise ConfigError(f"{where}: 'zerobias:' is given but add_zerobias is false")

    source = _zerobias_source(raw["zerobias"], f"{where}: zerobias") \
        if add_zerobias else None
    return OverlayConfig(_cleaning_cuts(raw["mc_clean"], f"{where}: mc_clean"), source)


def _cleaning_cuts(raw, where: str) -> CleaningCuts:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    _require_keys(raw, _CLEAN_KEYS, _CLEAN_KEYS, where)

    window = raw["csc_tpeak"]
    if not (isinstance(window, list) and len(window) == 2
            and all(_is_number(v) for v in window) and window[0] < window[1]):
        raise ConfigError(f"{where}: csc_tpeak must be [lo, hi] with lo < hi, "
                          f"got {window!r}")

    bxs = raw["rpc_bx"]
    if not (isinstance(bxs, list) and bxs
            and all(isinstance(v, int) and not isinstance(v, bool) for v in bxs)):
        raise ConfigError(f"{where}: rpc_bx must be a list of integers, got {bxs!r}")

    dr, genjet_pt = raw["dr"], raw["genjet_pt"]
    if not (_is_number(dr) and dr > 0):
        raise ConfigError(f"{where}: dr must be a positive number, got {dr!r}")
    if not (_is_number(genjet_pt) and genjet_pt >= 0):
        raise ConfigError(f"{where}: genjet_pt must be a number >= 0, got {genjet_pt!r}")

    return CleaningCuts(csc_tpeak=(window[0], window[1]), rpc_bx=tuple(bxs),
                        dr=dr, genjet_pt=genjet_pt)


def _zerobias_source(raw, where: str) -> ZeroBiasSource:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    _require_keys(raw, _ZEROBIAS_KEYS, _ZEROBIAS_KEYS, where)

    pool, lumi_mask = raw["pool"], raw["lumi_mask"]
    if not (isinstance(pool, str) and pool):
        raise ConfigError(f"{where}: pool must name a datasets.yaml entry, got {pool!r}")
    if not (isinstance(lumi_mask, str) and lumi_mask):
        raise ConfigError(f"{where}: lumi_mask must be a path, got {lumi_mask!r}")

    data_clean = raw["data_clean"]
    if data_clean is False:
        return ZeroBiasSource(pool, lumi_mask, None)
    if not isinstance(data_clean, dict):
        raise ConfigError(f"{where}: data_clean must be false or {{dr: <value>}}, "
                          f"got {data_clean!r}")
    _require_keys(data_clean, ("dr",), ("dr",), f"{where}: data_clean")
    dr = data_clean["dr"]
    if not (_is_number(dr) and dr > 0):
        raise ConfigError(f"{where}: data_clean dr must be a positive number, got {dr!r}")
    return ZeroBiasSource(pool, lumi_mask, dr)


def _require_keys(raw: dict, allowed, required, where: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; known: {list(allowed)}")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
