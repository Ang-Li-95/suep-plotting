"""The MDS derived columns: ``derive(events, settings) -> events``.

The processor calls this once per chunk, before filling histograms, and uses
the returned array in place of ``events``.  On MDSNANO samples (detected by the
presence of ``cscRechits``) it attaches:

``events.llp``
    One entry per generated LLP -- kinematics, decay vertex, decay-volume
    flags, per-system truth rechit counts, reconstruction flags and the
    eta-phi spread of its matched rechits.  Empty on samples without truth,
    so llp-based expressions still evaluate.
``events.cscCluster`` / ``events.dtCluster`` / ``events.rpcCluster``
    DBSCAN clusters of the muon-system rechits -- size, position, timing,
    shape moments, isolation from prompt objects, and the truth match.  Which
    collections exist depends on ``rpc_mode`` (no ``rpcCluster`` outside
    ``separate``).

Which of the steps below run, and with what parameters, comes from the config
set's ``columns.yaml`` -- see **docs/derived-columns.md** for the field-by-field
reference, the parameter list and what each step costs.  Nothing here reads the
environment or module state: :func:`configure` returns a
:class:`params.Settings` and the pipeline is handed it, so a run is reproducible
from its config directory alone and a worker process cannot lose it.

Module layout: this file holds only the pipeline (:func:`derive` and the
``_step_*`` functions) and re-exports the helpers, so ``import custom.columns``
still gives the whole interface.  ``params.py`` owns the schema, ``clustering.py``
the DBSCAN, ``llp.py`` the gen-level collection, ``isolation.py`` the cluster-to-
prompt-object dR.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# The settings and the helpers the pipeline below calls.
from .clustering import _cluster_merged, _cluster_system, _match_rpc
from .isolation import NO_OBJECT_DR, _dr_to_nearest, _selected_objects
from .llp import _build_llps, _empty_llps, _llp_rechit_dr
from .params import (DEFAULT_PARAMS, DEFAULTS, LLP_PDGID, PARAM_SPEC,
                     STEP_DEPS, Settings, _dbscan_params, configure)

# Re-exported so this module stays the single entry point for the package:
# notebooks, scripts/event_display.py and tests reach the helpers through
# ``custom.columns``, whichever file they now live in.
from .clustering import (  # noqa: F401
    SOURCE_CODES, _flat_chamber_code, _flat_layer_key, _flat_zlayer_key)
from .isolation import _jet_id, _jet_id_evaluator  # noqa: F401
from .llp import _dr_field_names, _dr_kernel, _llpidx_is_genpart_index  # noqa: F401


__all__ = ["derive", "configure", "Settings", "DEFAULTS", "DEFAULT_PARAMS",
           "PARAM_SPEC", "STEP_DEPS", "NO_OBJECT_DR", "LLP_PDGID"]


# The cluster collections, in the order they are built and attached; with
# rpc_mode merge/match there is no "rpc" entry (its rechits belong to the
# other two).
_CLUSTER_SYSTEMS = (("csc", "CSC"), ("dt", "DT"), ("rpc", "RPC"))


def _hit_fields(SYS, settings):
    """The ``llp.nHits*`` fields one cluster system's rechits are counted in.

    Only ``rpc_mode: merge`` puts RPC rechits *inside* a cluster, so only there
    do they belong in the denominator of ``clusterHitFrac``; under ``match``
    they sit next to the cluster and are counted by ``nRPCHits``.
    """
    if settings["rpc_mode"] != "merge":
        return (SYS,)
    return {"CSC": ("CSC", "RPCEndcap"), "DT": ("DT", "RPCBarrel")}[SYS]


class _Context:
    """Work area the steps hand to each other (see :func:`derive`).

    ``events`` and ``llp`` are rebuilt by ``ak.with_field``, so the steps
    reassign them on the context rather than mutating arrays in place.

    ``settings`` is the resolved :class:`params.Settings` for *this* call.  The
    steps read it off the context and pass it down to the helpers; no module
    state is involved, so a process that was handed a config cannot lose it on
    the way to the code that acts on it, and two configurations can run side by
    side (an eps scan in a notebook) without one clobbering the other.
    """

    def __init__(self, events, settings):
        self.settings = settings
        self.steps = settings.steps
        self.events = events
        self.has_truth = "SUEPGenPart" in events.fields
        self.clusters = {}
        self.llp = None
        # Per-event LLP identifiers the rechit ``llpIdx`` branches are compared
        # against; set by the ``llp`` step (see _step_llp).
        self.match_key = None


def derive(events, settings=None):
    """Attach the MDS LLP/cluster collections (MDSNANO samples only).

    Runs the optional helpers selected by :func:`configure` (all of them by
    default) and attaches whatever they produced.  Fields belonging to a
    disabled step are deliberately absent, so a config referencing them fails
    at expression validation rather than silently filling zeros.

    *settings* is a :class:`params.Settings` from :func:`configure`; omitting
    it runs on the defaults.  Nothing is read from module state, so the caller
    -- the processor, a notebook, a test -- always knows what it configured.
    """
    if "cscRechits" not in events.fields:
        return events

    ctx = _Context(events, DEFAULTS if settings is None else settings)
    for step in ctx.steps:
        _STEP_FUNCS[step](ctx)

    events = ctx.events
    if ctx.llp is not None:
        events = ak.with_field(events, ctx.llp, "llp")
    for sys, _ in _CLUSTER_SYSTEMS:
        if sys in ctx.clusters:
            events = ak.with_field(events, ctx.clusters[sys], sys + "Cluster")
    return events


def _step_clusters(ctx):
    """DBSCAN the rechit systems (~35% of derive()).

    Three layouts, selected by the ``rpc_mode`` parameter: the three systems on
    their own ("separate"), the RPC rechits folded into the system they overlap
    ("merge", see :func:`_merged_clusters`), or the same CSC/DT clustering as
    "separate" with the RPC rechits associated to the finished clusters
    afterwards ("match", see :func:`_matched_clusters`).
    """
    mode = ctx.settings["rpc_mode"]
    if mode == "merge":
        _merged_clusters(ctx)
        return
    systems = (("csc", "cscRechits", "Tpeak"), ("dt", "dtRecHits", None))
    if mode == "separate":
        systems += (("rpc", "rpcRecHits", "Time"),)
    for sys, coll, timefield in systems:
        eps, min_samples = _dbscan_params(sys, ctx.settings)
        ctx.clusters[sys] = _cluster_system(ctx.events[coll], timefield,
                                            eps, min_samples, ctx.settings, sys)
    if mode == "match":
        _matched_clusters(ctx)


def _rpc_regions(events):
    """The RPC rechits split into (barrel, endcap) by their ``Region`` branch.

    ``Region`` is 0 in the barrel and +-1 in the two endcaps, so the barrel
    wheels pair with DT and the endcap disks with CSC -- the two are at the
    same radii / z as the system they are read out alongside.
    """
    rpc = events.rpcRecHits
    return rpc[rpc.Region == 0], rpc[rpc.Region != 0]


def _merged_clusters(ctx):
    """DBSCAN with the RPC rechits merged into the overlapping system.

    Barrel RPC clusters with DT, endcap RPC with CSC, so a shower crossing
    both leaves one cluster instead of two that later have to be paired up.
    No ``rpcCluster`` collection is produced: every RPC rechit has been offered
    to the clustering of the system it overlaps, and clustering it a second
    time on its own would double count it.  The RPC hits a cluster picked up
    are what date it (the ``rpcTime*`` / ``rpcBx*`` fields) -- DT rechits carry
    no time branch at all, and the CSC ``Tpeak`` is on its own clock.
    """
    barrel, endcap = _rpc_regions(ctx.events)
    eps, min_samples = _dbscan_params("csc", ctx.settings)
    ctx.clusters["csc"] = _cluster_merged(
        [(ctx.events.cscRechits, "Tpeak", "csc"), (endcap, "Time", "rpc")],
        eps, min_samples, ctx.settings)
    eps, min_samples = _dbscan_params("dt", ctx.settings)
    ctx.clusters["dt"] = _cluster_merged(
        [(ctx.events.dtRecHits, None, "dt"), (barrel, "Time", "rpc")],
        eps, min_samples, ctx.settings)


def _matched_clusters(ctx):
    """Date the finished CSC/DT clusters with the RPC rechits around them.

    The counterpart of :func:`_merged_clusters`: same RPC fields on the same
    two collections, but the clustering never saw an RPC rechit, so the
    cluster variables are bit-identical to a ``rpc_mode: separate`` run and
    only the RPC annotation is new.  See :func:`clustering._match_rpc`.
    """
    barrel, endcap = _rpc_regions(ctx.events)
    for sys, rpc in (("csc", endcap), ("dt", barrel)):
        eps, _ = _dbscan_params(sys, ctx.settings)
        ctx.clusters[sys] = _match_rpc(ctx.clusters[sys], rpc, eps, sys)


def _step_jerc(ctx):
    """JEC on events.Jet, in place; plus JER smearing on MC.

    JEC/JER rescale pT and mass and leave eta/phi alone, so this reaches the
    cluster dR only through the ``iso_jet_pt`` threshold -- but the threshold is
    exactly what the jet selection is for, so the correction has to come first.

    Data is corrected too, with the era's ``*_DATA`` tag: ``L1L2L3Res`` carries
    the residual corrections that exist precisely for data.  Only the JER
    smearing is MC-only.  ``genWeight`` is the MC marker (``has_truth`` is False
    for the non-SUEP MC too, so it cannot be used here).
    """
    if not ctx.settings["jerc"] or "Jet" not in ctx.events.fields:
        return

    from suep_plot.jme import DEFAULTS, correct_jets

    era, algo = ctx.settings["jerc_era"], ctx.settings["jerc_algo"]
    if "genWeight" in ctx.events.fields:
        ctx.events = correct_jets(ctx.events, era=era, algo=algo)
        return

    tag = ctx.settings["jerc_data_tag"] or DEFAULTS.get(era, {}).get("jec_tag_data")
    if tag is None:
        raise ValueError(
            f"columns.yaml: jerc is on and this is data, but era '{era}' has no "
            "jec_tag_data in suep_plot.jme.DEFAULTS -- set jerc_data_tag "
            "explicitly, or jerc: false to leave data jets uncorrected.")
    ctx.events = correct_jets(ctx.events, era=era, algo=algo,
                              jec_tag=tag, smear=False)


def _step_cluster_isolation(ctx):
    """dR from each cluster centroid to the nearest selected prompt object.

    One dR field per entry of the ``iso_objects`` parameter, so which objects
    count -- and how they are selected -- lives entirely in the config set.
    """
    objects = {field: _selected_objects(ctx.events, selection, ctx.settings)
               for field, selection in ctx.settings["iso_objects"].items()}

    for sys, clusters in ctx.clusters.items():
        for field, objs in objects.items():
            if objs is not None:
                clusters = ak.with_field(
                    clusters, _dr_to_nearest(clusters, objs), field)
        ctx.clusters[sys] = clusters


def _step_llp(ctx):
    """The ``events.llp`` collection (empty on samples without truth)."""
    if not ctx.has_truth:
        ctx.llp = _empty_llps(ctx.events, ctx.settings)
        return
    ctx.llp = _build_llps(ctx.events)
    # The rechit truth branch ``llpIdx`` uses one of two conventions depending
    # on the producer: the SUEPGenPart index of the LLP (post-Geant4-fix,
    # SUEPs_Gen2; the default) or the ordinal LLP index within the event
    # (pre-fix, SUEPs_Gen; llpidx_convention: ordinal).  Match against the LLP
    # field in the same space.
    ctx.match_key = (ctx.llp.gidx
                     if ctx.settings["llpidx_convention"] == "genpart"
                     else ctx.llp.lidx)


def _step_llp_hits(ctx):
    """Per-LLP truth rechit counts, per system and pooled."""
    if not ctx.has_truth:
        return               # _empty_llps already carries the fields
    barrel, endcap = _rpc_regions(ctx.events)
    nhits = {}
    for sys, rechits in (("CSC", ctx.events.cscRechits),
                         ("DT", ctx.events.dtRecHits),
                         ("RPC", ctx.events.rpcRecHits),
                         # The two RPC regions on their own: the denominators
                         # of the merged clustering (see _step_llp_reco).
                         ("RPCBarrel", barrel), ("RPCEndcap", endcap)):
        nhits[sys] = ak.values_astype(
            ak.sum(ctx.match_key[:, :, None] == rechits.llpIdx[:, None, :], axis=2),
            np.int64)
    llp = ctx.llp
    for sys in ("CSC", "DT", "RPC", "RPCBarrel", "RPCEndcap"):
        llp = ak.with_field(llp, nhits[sys], "nHits" + sys)
    ctx.llp = ak.with_field(llp, nhits["CSC"] + nhits["DT"] + nhits["RPC"], "nHitsTotal")


def _step_llp_reco(ctx):
    """Reconstruction flags: LLP has a truth-matched DBSCAN cluster."""
    if not ctx.has_truth:
        return               # _empty_llps already carries the fields
    llp, reco = ctx.llp, {}
    for sys, SYS in _CLUSTER_SYSTEMS:
        if sys not in ctx.clusters:
            continue
        # (ev, nLLP, nCluster) truth-match table: cluster's best-LLP index
        # equals this LLP's.  Unmatched clusters carry -1, which never equals
        # an LLP index (>= 0), so they drop out.
        sel = ctx.match_key[:, :, None] == ctx.clusters[sys].matchedLLPIdx[:, None, :]
        reco[sys] = ak.any(sel, axis=2)
        # (2) number of DBSCAN clusters truth-matched to each LLP
        nclu = ak.values_astype(ak.sum(sel, axis=2), np.int64)
        llp = ak.with_field(llp, nclu, "nRecoCluster" + SYS)
        # (3) fraction of the LLP's matched rechits captured by its best
        # matched cluster: max over matched clusters of that cluster's
        # matched-hit count, divided by the LLP's total matched rechits
        # in the system.  NaN when the LLP has no matched rechit there.
        hits_in = ak.where(sel, ctx.clusters[sys].nMatchedHits[:, None, :], 0)
        best = ak.fill_none(ak.max(hits_in, axis=2), 0)
        # With rpc_mode: merge the cluster also holds the RPC rechits of the
        # region it covers, so those count towards the LLP's hits in "this
        # system" (see _hit_fields; under "match" they do not).
        nh = sum(llp["nHits" + name] for name in _hit_fields(SYS, ctx.settings))
        frac = ak.where(nh > 0, best / ak.where(nh > 0, nh, 1), np.nan)
        llp = ak.with_field(llp, frac, "clusterHitFrac" + SYS)
    for sys, SYS in _CLUSTER_SYSTEMS:
        if sys in reco:
            llp = ak.with_field(llp, reco[sys], "reco" + SYS)
    any_reco = None
    for flag in reco.values():
        any_reco = flag if any_reco is None else (any_reco | flag)
    ctx.llp = ak.with_field(llp, any_reco, "reco")


def _step_llp_shape(ctx):
    """Eta-phi spread of each LLP's truth-matched rechits.

    Per system and for the three pooled ("Total", matching nHitsTotal).  The
    per-system passes also write each matched rechit's dR to its LLP's centroid
    back onto the rechit collection (``drLLP``, plus the ``matchedLLP`` flag).
    Samples without truth branches get the same fields (all NaN / False), so
    the configs evaluate on background as well.
    """
    events, llp = ctx.events, ctx.llp
    nllp = ak.num(llp.pt)
    for sys, coll in (("CSC", "cscRechits"), ("DT", "dtRecHits"),
                      ("RPC", "rpcRecHits")):
        rechits = events[coll]
        counts = ak.num(rechits.Eta)
        hit_dr = np.full(int(ak.sum(counts)), np.nan)
        if ctx.has_truth:
            for f, v in _llp_rechit_dr(events, [coll], ctx.match_key,
                                       ctx.settings, hit_dr=hit_dr).items():
                llp = ak.with_field(llp, ak.unflatten(v, nllp), f + sys)
        matched = (rechits.llpIdx >= 0 if "llpIdx" in rechits.fields
                   else ak.values_astype(ak.zeros_like(rechits.Eta), np.bool_))
        rechits = ak.with_field(rechits, ak.unflatten(hit_dr, counts), "drLLP")
        events = ak.with_field(events, ak.with_field(rechits, matched, "matchedLLP"),
                               coll)
    if ctx.has_truth:
        for f, v in _llp_rechit_dr(
                events, ["cscRechits", "dtRecHits", "rpcRecHits"],
                ctx.match_key, ctx.settings).items():
            llp = ak.with_field(llp, ak.unflatten(v, nllp), f + "Total")
    ctx.events, ctx.llp = events, llp


_STEP_FUNCS = {
    "jerc": _step_jerc,
    "clusters": _step_clusters,
    "cluster_isolation": _step_cluster_isolation,
    "llp": _step_llp,
    "llp_hits": _step_llp_hits,
    "llp_reco": _step_llp_reco,
    "llp_shape": _step_llp_shape,
}
assert set(_STEP_FUNCS) == set(STEP_DEPS)
