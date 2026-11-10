"""DBSCAN clustering of the muon-system rechits.

:func:`_cluster_system` turns one rechit collection into a jagged record array
of clusters (dR metric in eta-phi, per-cluster shape/position/truth-match
fields).  :func:`_cluster_merged` does the same for *several* collections
clustered together -- the RPC-merged configuration, where the barrel RPC
rechits join the DT ones and the endcap RPC rechits join the CSC ones, so that
an LLP shower crossing both systems becomes a single cluster.  The
``_flat_*_key`` helpers build the flat integer ids it counts hits over: the
depth segment ("layer"), the chamber-type code, and the layer plane read off
the quantized global z.
"""

from __future__ import annotations

import awkward as ak
import numpy as np


# Which detector a merged cluster's hit came from (the ``source`` per-hit array
# and the per-cluster ``firstSystem`` field).
SOURCE_CODES = {"csc": 0, "dt": 1, "rpc": 2}

# Offsets that keep the flat ids of two merged systems from colliding: a DT
# layer key and an RPC layer key are both small integers, and a DT and an RPC
# chamber code can be identical.  Applied to the RPC component of a merge and
# to nothing else -- RPC is never the primary system (it is merged *into* CSC
# or DT), so CSC and DT keep their historical ids, and the shift means the same
# thing in cscCluster and dtCluster instead of depending on the order the
# components were passed in.
_LAYER_KEY_STRIDE = 10 ** 12
_CHAMBER_CODE_STRIDE = 1000


def _flat_layer_key(rechits):
    """Flat int64 id of the finest depth segment ("layer") per system.

    CSC has no in-chamber layer branch in MDSNano, so its layer unit is the
    physical chamber; DT/RPC use the stored physical layer.  See
    :func:`_flat_zlayer_key` for the complementary global-z definition.
    """
    def flat(field):
        return np.asarray(ak.flatten(rechits[field]), dtype=np.int64)

    f = rechits.fields
    if "IChamber" in f:      # CSC: chamber type (+-11..42) x chamber number
        return (flat("Chamber") + 50) * 100 + flat("IChamber")
    if "SuperLayer" in f:    # DT: wheel/sector/station/superlayer/layer
        return ((((flat("Wheel") + 2) * 15 + flat("Sector")) * 5
                 + flat("Station")) * 20
                + (flat("SuperLayer") - 1) * 4 + flat("Layer"))
    # RPC: region/ring/sector/station/layer
    return (((((flat("Region") + 2) * 8 + flat("Ring") + 2) * 20
              + flat("Sector")) * 5 + flat("Station")) * 3 + flat("Layer"))


def _flat_chamber_code(rechits):
    """Flat int64 chamber-type code of every rechit, as ``station * 10 + ring``.

    Read off the innermost hit of a cluster this identifies where the cluster
    starts, which is what separates a genuine shower from a punch-through: a jet
    that leaks out of the calorimeter enters the muon system at its innermost
    chamber (ME1/1, ME1/2 for CSC), so a cluster whose first hit sits there is
    suspect.

    CSC stores the code directly as ``Chamber`` (signed for the two endcaps;
    the sign is dropped so both share a bin).  DT and RPC have no such branch,
    so the code is built from the station and the ring-like coordinate -- the
    wheel for DT, the ring for RPC -- in the same ``station * 10 + ring``
    layout.  RPC additionally offsets the endcap by 100, since a barrel and an
    endcap chamber can otherwise share a code.
    """
    def flat(field):
        return np.abs(np.asarray(ak.flatten(rechits[field]), dtype=np.int64))

    f = rechits.fields
    if "IChamber" in f:                                # CSC: ME<station>/<ring>
        return flat("Chamber")
    if "SuperLayer" in f:                              # DT: MB<station>/<|wheel|>
        return flat("Station") * 10 + flat("Wheel")
    return (flat("Station") * 10 + flat("Ring")        # RPC: RB/RE<station>/<ring>
            + 100 * flat("Region"))


def _flat_zlayer_key(rechits):
    """Flat int64 id of the layer plane from the global rechit z (1 mm bins).

    CSC layer planes are normal to the beam, so quantized global z identifies
    the in-chamber layer (6 discrete planes per chamber, 2.54 cm spacing,
    2.2 cm in ME1/1; 132 planes in total) -- finer than the chamber unit of
    :func:`_flat_layer_key`.  CAVEAT: the central MDSNano v2 background
    production stores ONE z value per chamber (in-chamber layer information
    is dropped), so the ``zLayer*`` cluster fields are only comparable
    between samples whose ntuples keep the true per-layer z (the private
    signal production does; a reprocessed background sample would).  Only
    meaningful for CSC -- for barrel systems z runs along the wires, not
    through the layers.
    """
    return np.round(
        np.asarray(ak.flatten(rechits.Z), dtype=np.float64) * 10.0
    ).astype(np.int64)


def _flat_hits(rechits, timefield, system):
    """Materialize one rechit collection as flat numpy arrays.

    Per-event awkward indexing is slow, so every branch the clustering needs is
    flattened once per chunk and events are then processed as offset slices.
    The returned dict is what :func:`_merge_hits` concatenates and
    :func:`_cluster_hits` clusters; ``system`` is the key of
    :data:`SOURCE_CODES` this collection belongs to.
    """
    counts = np.asarray(ak.num(rechits.Eta))
    n = int(counts.sum())

    def flat(field, dtype=np.float64):
        return np.asarray(ak.flatten(rechits[field]), dtype=dtype)

    xs, ys, zs = flat("X"), flat("Y"), flat("Z")
    fields = rechits.fields
    has_truth = "llpIdx" in fields
    return {
        "counts": counts,
        "eta": flat("Eta"),
        "phi": flat("Phi"),
        "x": xs,
        "y": ys,
        "z": zs,
        # Distance to the nominal interaction point, to pick a cluster's
        # first hit.
        "dist": np.sqrt(xs ** 2 + ys ** 2 + zs ** 2),
        "station": np.asarray(ak.flatten(rechits.Station)),
        "layerkey": _flat_layer_key(rechits),
        "zlayerkey": _flat_zlayer_key(rechits),
        "chamber": _flat_chamber_code(rechits),
        "llpidx": (np.asarray(ak.flatten(rechits.llpIdx)) if has_truth
                   else np.full(n, -1, dtype=np.int64)),
        "system": system,
        "code": SOURCE_CODES[system],
        "source": np.full(n, SOURCE_CODES[system], dtype=np.int64),
        "time": (flat(timefield) if timefield is not None
                 else np.full(n, np.nan)),
        # RPC-only timing extras; NaN everywhere else, so the merged arrays
        # have one dtype and the RPC-time estimate below just masks on source.
        "timeError": (flat("TimeError") if "TimeError" in fields
                      else np.full(n, np.nan)),
        "bx": (flat("Bx") if "Bx" in fields else np.full(n, np.nan)),
        "has_time": timefield is not None,
        "has_truth": has_truth,
    }


# Per-hit arrays of _flat_hits that _merge_hits concatenates (the rest are
# scalars or the per-event counts).
_HIT_ARRAYS = ("eta", "phi", "x", "y", "z", "dist", "station", "layerkey",
               "zlayerkey", "chamber", "llpidx", "source", "time", "timeError",
               "bx")


def _merge_hits(components):
    """Interleave several :func:`_flat_hits` dicts into one, event by event.

    The clustering runs on per-event slices of flat arrays, so the merged
    arrays have to be ordered by event first and by source collection second
    -- not simply concatenated, which would put all of one system's hits
    before all of the other's.  The destination index of every hit is built
    vectorized, then each array is scattered into place.

    The layer/chamber ids of the RPC component are offset so that two systems
    cannot share an id (a DT and an RPC chamber code are otherwise the same
    small integer).  Keyed on the system, not on the position in *components*:
    RPC is always the one merged in, never the primary, so CSC and DT keep the
    ids they have always had -- a cluster's ``firstChamber`` still reads as the
    plain CSC/DT chamber code and only an RPC-first cluster stands out -- and
    ``+_CHAMBER_CODE_STRIDE`` means "RPC" in both merged collections rather
    than "whichever component came second".  A single component is returned
    as-is, which keeps single-system clustering bit-identical to before.
    """
    if len(components) == 1:
        return components[0]

    # One offset per component, so a collision would silently merge two
    # physically different chambers into one id.
    shifts = [int(c["code"] == SOURCE_CODES["rpc"]) for c in components]
    if len(set(shifts)) != len(shifts):
        raise ValueError(
            "clustering: cannot merge " + ", ".join(c["system"] for c in components)
            + " -- the id offsets separate RPC from the system it is merged "
              "into, so at most one non-RPC component can go in")

    counts = np.sum([c["counts"] for c in components], axis=0)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(counts.sum())

    merged = {name: np.empty(total, dtype=np.result_type(
        *[c[name].dtype for c in components])) for name in _HIT_ARRAYS}

    # Running per-event write position: component k starts after the hits of
    # components 0..k-1 in the same event.
    start = offsets[:-1].copy()
    for shift, comp in zip(shifts, components):
        c_counts = comp["counts"]
        c_offsets = np.concatenate([[0], np.cumsum(c_counts)])
        rank = np.arange(int(c_counts.sum())) - np.repeat(c_offsets[:-1], c_counts)
        dest = np.repeat(start, c_counts) + rank
        for name in _HIT_ARRAYS:
            values = comp[name]
            if name in ("layerkey", "zlayerkey"):
                values = values + shift * _LAYER_KEY_STRIDE
            elif name == "chamber":
                values = values + shift * _CHAMBER_CODE_STRIDE
            merged[name][dest] = values
        start = start + c_counts

    merged["counts"] = counts
    merged["system"] = components[0]["system"]
    merged["code"] = components[0]["code"]
    # The merged collection has a time if its *primary* (first) component does:
    # that is the field the historical ``time`` means -- CSC Tpeak.  The RPC
    # hits are dated separately, by the rpcBx* fields.
    merged["has_time"] = components[0]["has_time"]
    merged["has_truth"] = all(c["has_truth"] for c in components)
    return merged


# The RPC fields a cluster carries, in both rpc_mode: merge and rpc_mode: match
# -- the same names and the same meanings, so one config set plots either.
RPC_CLUSTER_FIELDS = ("firstSystem", "nRPCHits", "nRPCHitsBx0", "rpcHitFrac",
                      "rpcBx", "rpcBxMedian", "rpcBxSpread", "rpcOutOfTimeFrac")


def _match_rpc(clusters, rpc, eps, system):
    """Attach the RPC timing of the rechits *near* each cluster (rpc_mode: match).

    The alternative to merging: the clustering stays exactly the single-system
    one, and the RPC rechits are associated to the finished clusters
    afterwards -- every hit within *eps* in dR of a cluster centroid, assigned
    to its nearest cluster so no hit dates two showers.

    RPC then contributes nothing to the DBSCAN density estimate, which is the
    point.  RPC rechit times are flat in BX in data (only ~19% of the rechits
    of a ZeroBias event sit at BX 0), so merged, RPC noise can push a
    background cluster over ``min_samples``; matched, it can only annotate a
    cluster the CSC or DT rechits already made on their own.  The cluster
    variables therefore keep the values -- and the signal region keeps the
    definition -- of the standard MDS analysis.

    The fields are :data:`RPC_CLUSTER_FIELDS`, the same ones a merged cluster
    carries.  ``rpcHitFrac`` means the same thing in both modes, the RPC share
    of the cluster's hits: matching leaves the RPC hits out of ``size``, so it
    is n / (size + n) here and n / size there.  ``firstSystem`` is always the
    primary system, since no RPC hit is part of the cluster -- so the
    ``first_not_rpc`` selections simply pass everything, rather than needing a
    config of their own.
    """
    hits = _flat_hits(rpc, "Time", "rpc")
    code = SOURCE_CODES[system]

    nclu = np.asarray(ak.num(clusters.eta))
    clu_off = np.concatenate([[0], np.cumsum(nclu)])
    ceta = np.asarray(ak.flatten(clusters.eta), dtype=np.float64)
    cphi = np.asarray(ak.flatten(clusters.phi), dtype=np.float64)
    csize = np.asarray(ak.flatten(clusters.size), dtype=np.float64)
    hit_off = np.concatenate([[0], np.cumsum(hits["counts"])])

    out = {f: [] for f in RPC_CLUSTER_FIELDS}
    empty = np.zeros(0, dtype=np.float64)
    for i in range(len(nclu)):
        c = slice(int(clu_off[i]), int(clu_off[i + 1]))
        if c.start == c.stop:
            continue
        h = slice(int(hit_off[i]), int(hit_off[i + 1]))
        ce, cp = ceta[c], cphi[c]
        owner = None
        if h.stop > h.start:
            deta = hits["eta"][h][None, :] - ce[:, None]
            dphi = hits["phi"][h][None, :] - cp[:, None]
            dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
            dr = np.sqrt(deta ** 2 + dphi ** 2)
            # Nearest cluster wins, so a hit between two clusters dates one.
            nearest = dr.argmin(axis=0)
            owner = np.where(dr.min(axis=0) < eps, nearest, -1)
        for j in range(c.stop - c.start):
            m = owner == j if owner is not None else np.zeros(0, dtype=bool)
            n = int(m.sum())
            bx = hits["bx"][h][m] if n else empty
            out["firstSystem"].append(code)
            out["nRPCHits"].append(n)
            out["nRPCHitsBx0"].append(int(np.count_nonzero(bx == 0)))
            out["rpcHitFrac"].append(float(n / (csize[c][j] + n)) if n else 0.0)
            for f, v in _rpc_time(bx).items():
                out[f].append(v)

    ints = {"firstSystem", "nRPCHits", "nRPCHitsBx0"}
    for f in RPC_CLUSTER_FIELDS:
        values = np.asarray(out[f], dtype=np.int64 if f in ints else np.float64)
        clusters = ak.with_field(clusters, ak.unflatten(values, nclu), f)
    return clusters


def _rpc_time(bx):
    """Timing of one cluster's RPC hits, from the bunch crossing.

    RPC is the only muon subdetector whose MDSNano rechits carry any timing at
    all, so its hits are what date a CSC or DT cluster.  The estimate is the
    bunch crossing: mean / median / RMS in BX units (multiply by 25 ns for a
    time), plus ``rpcOutOfTimeFrac``, the fraction of the cluster's RPC hits
    outside the in-time BX -- the handle on a late, displaced shower.

    The rechit *time* is deliberately not used.  No MDSNano production writes
    it: ``Time`` is exactly 0 and ``TimeError`` exactly -1 for every RPC rechit
    in every sample checked (private signal, central DY, ZeroBias data -- 841k
    rechits, one distinct value each), so a mean, median, spread or
    inverse-variance weighted time computed from it says nothing.  The BX,
    by contrast, is filled and spans -3..4.  If a production ever fills the
    time, add the fields back here and in configs/common/histograms.yaml.

    A cluster with no RPC hit at all gives NaN throughout.
    """
    fields = ("rpcBx", "rpcBxMedian", "rpcBxSpread", "rpcOutOfTimeFrac")
    out = dict.fromkeys(fields, np.nan)

    good_bx = bx[np.isfinite(bx)]
    if len(good_bx):
        out["rpcBx"] = float(good_bx.mean())
        out["rpcBxMedian"] = float(np.median(good_bx))
        out["rpcBxSpread"] = float(good_bx.std())
        out["rpcOutOfTimeFrac"] = float((good_bx != 0).mean())
    return out


def dbscan_labels(eta, phi, eps, min_samples):
    """DBSCAN labels of one event's rechits (-1 = noise), in the eta-phi dR metric.

    The one implementation of the per-event clustering: :func:`_cluster_hits`
    builds its cluster records from these labels, and scripts/event_display.py
    colours the hits by them, so the display shows exactly the analysis's
    clusters.  The distance matrix is precomputed to handle the phi
    wrap-around (see docs/known-issues.md for its memory cost).
    """
    from sklearn.cluster import DBSCAN

    if len(eta) < min_samples:
        return np.full(len(eta), -1, dtype=np.int64)
    dphi = phi[:, None] - phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dmat = np.sqrt((eta[:, None] - eta[None, :]) ** 2 + dphi ** 2)
    return DBSCAN(eps=eps, min_samples=min_samples,
                  metric="precomputed").fit_predict(dmat)


# ── the fields of one cluster ──────────────────────────────────────
# Each function takes the arrays of one cluster's hits and returns the fields
# it computes; _cluster_hits puts them together.  The names below are the
# record's fields, in order.

_POSITION_FIELDS = ("size", "eta", "phi", "x", "y", "z", "r", "etaSpread",
                    "phiSpread", "rSpread", "zSpread")
_STATION_FIELDS = ("nStation", "stationSpan", "avgStation", "maxStationFrac")
_FIRST_HIT_FIELDS = ("firstChamber", "firstStation")
_TRUTH_FIELDS = ("nMatchedHits", "matchedLLPIdx", "matched", "nMatchedLLP", "hasTruth")
_TIME_FIELDS = ("time", "timeSpread", "ootHitFrac")

# Fields that are not float64.
_FIELD_DTYPES = {
    "size": np.int64, "nStation": np.int64, "stationSpan": np.int64,
    "nLayer": np.int64, "nZLayer": np.int64, "nMatchedHits": np.int64,
    "firstChamber": np.int64, "firstStation": np.int64,
    "matchedLLPIdx": np.int64, "matched": np.bool_,
    "nMatchedLLP": np.int64, "hasTruth": np.bool_,
    "firstSystem": np.int64, "nRPCHits": np.int64, "nRPCHitsBx0": np.int64,
}


def _unit_fields(unit: str) -> tuple:
    """Field names of :func:`_hits_per_unit` for ``Layer`` or ``ZLayer``."""
    low = unit[0].lower() + unit[1:]
    return (f"n{unit}", f"{low}HitsMean", f"{low}HitsRMS", f"{low}HitsRelRMS",
            f"max{unit}Frac")


def _cluster_fields(has_time: bool, merged: bool) -> list:
    """The per-cluster scalar fields, in the order of the cluster record."""
    fields = [*_POSITION_FIELDS, *_STATION_FIELDS, *_unit_fields("Layer"),
              *_unit_fields("ZLayer"), *_FIRST_HIT_FIELDS, *_TRUTH_FIELDS]
    if has_time:
        fields += _TIME_FIELDS
    if merged:
        fields += RPC_CLUSTER_FIELDS
    return fields


def _position(eta, phi, x, y, z) -> dict:
    """Size, centroid and spread of a cluster.

    phi is averaged on the circle, so hits either side of +-pi give a centroid
    between them and not at 0.
    """
    cphi = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
    dphi = (phi - cphi + np.pi) % (2 * np.pi) - np.pi
    return {
        "size": len(eta),
        "eta": float(eta.mean()),
        "phi": float(cphi),
        "x": float(x.mean()),
        "y": float(y.mean()),
        "z": float(z.mean()),
        "r": float(np.hypot(x.mean(), y.mean())),
        "etaSpread": float(eta.std()),
        "phiSpread": float(np.sqrt(np.mean(dphi ** 2))),
        "rSpread": float(np.hypot(x, y).std()),
        "zSpread": float(z.std()),
    }


def _stations(station) -> dict:
    """Hit multiplicity across stations.

    A shower is contained in one station, a punch-through muon crosses several.
    """
    st = np.abs(station)
    _, counts = np.unique(st, return_counts=True)
    return {
        "nStation": len(counts),
        "stationSpan": int(st.max() - st.min()),
        "avgStation": float(st.mean()),
        "maxStationFrac": float(counts.max() / counts.sum()),
    }


def _hits_per_unit(keys, unit: str) -> dict:
    """How the hits spread over depth units: moments of the hits-per-unit count.

    A muon leaves about one hit per layer (uniform), a shower a strongly
    varying number ("fuzziness").  *unit* is ``Layer`` (the ids of
    :func:`_flat_layer_key`) or ``ZLayer`` (the layer plane from quantized
    global z, see :func:`_flat_zlayer_key` for its caveat).
    """
    _, counts = np.unique(keys, return_counts=True)
    mean = float(counts.mean())
    names = _unit_fields(unit)
    return dict(zip(names, (len(counts), mean, float(counts.std()),
                            float(counts.std()) / mean,
                            float(counts.max() / counts.sum()))))


def _first_hit(dist, chamber, station) -> tuple[dict, int]:
    """Where the cluster starts: its hit closest to the interaction point.

    A punch-through begins at an innermost chamber.  Also returns that hit's
    index within the cluster.
    """
    first = int(np.argmin(dist))
    return {"firstChamber": int(chamber[first]),
            "firstStation": int(abs(station[first]))}, first


_NO_LLP = np.zeros(0, dtype=np.int64)


def _truth_match(llpidx, match_min_hits: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Which LLPs the cluster's hits come from.

    The scalar fields name the dominant LLP (``matchedLLPIdx``, -1 unless it
    clears *match_min_hits*) and how many distinct LLPs clear the threshold
    (``nMatchedLLP``): a cluster can overlap more than one LLP's shower.

    Also returns *which* LLPs clear it and their hit counts.  The dominant one
    alone would make a second LLP whose shower landed in the same cluster look
    unreconstructed even though its hits are right there.
    """
    idx = llpidx[llpidx >= 0]
    if len(idx):
        uniq, counts = np.unique(idx, return_counts=True)
        best, nbest = int(uniq[counts.argmax()]), int(counts.max())
        clears = counts >= match_min_hits
        n_clearing = int(clears.sum())
        all_idx = uniq[clears].astype(np.int64)
        all_hits = counts[clears].astype(np.int64)
    else:
        best, nbest, n_clearing = -1, 0, 0
        all_idx = all_hits = _NO_LLP
    matched = nbest >= match_min_hits
    return {"nMatchedHits": nbest,
            "matchedLLPIdx": best if matched else -1,
            "matched": matched,
            "nMatchedLLP": n_clearing}, all_idx, all_hits


def _hit_time(time, oot_cut: float) -> dict:
    """Mean, spread and out-of-time fraction of the cluster's rechit times.

    The spread separates a real shower (every hit from the same particle, so
    one time) from a cluster DBSCAN built out of unrelated in-time and
    out-of-time hits; ``ootHitFrac`` says how much of it sits outside
    +-*oot_cut*.  NaN throughout if no hit has a time.
    """
    t = time[np.isfinite(time)]
    if not len(t):
        return dict.fromkeys(_TIME_FIELDS, np.nan)
    return {"time": float(t.mean()),
            "timeSpread": float(t.std()),
            "ootHitFrac": float((np.abs(t) > oot_cut).mean())}


def _rpc_content(source, bx, first: int) -> dict:
    """The RPC hits a merged cluster picked up (:data:`RPC_CLUSTER_FIELDS`)."""
    is_rpc = source == SOURCE_CODES["rpc"]
    rpc_bx = bx[is_rpc]
    return {"firstSystem": int(source[first]),
            "nRPCHits": int(is_rpc.sum()),
            "nRPCHitsBx0": int(np.count_nonzero(rpc_bx == 0)),
            "rpcHitFrac": float(is_rpc.sum() / len(source)),
            **_rpc_time(rpc_bx)}


def _one_cluster(c: dict, has_truth: bool, has_time: bool, merged: bool,
                 primary_code: int, settings) -> tuple[dict, np.ndarray, np.ndarray]:
    """Every field of one cluster, from the arrays *c* of its hits."""
    row = _position(c["eta"], c["phi"], c["x"], c["y"], c["z"])
    row.update(_stations(c["station"]))
    row.update(_hits_per_unit(c["layerkey"], "Layer"))
    row.update(_hits_per_unit(c["zlayerkey"], "ZLayer"))
    first_hit, first = _first_hit(c["dist"], c["chamber"], c["station"])
    row.update(first_hit)
    truth, all_idx, all_hits = _truth_match(c["llpidx"], settings["match_min_hits"])
    row.update(truth, hasTruth=has_truth)
    if has_time:
        # ``time`` is the primary system's time only (CSC Tpeak): the RPC hits
        # a merged cluster picked up are on a different clock and are dated
        # apart, by the rpcBx* fields.
        time = c["time"][c["source"] == primary_code] if merged else c["time"]
        row.update(_hit_time(time, settings["oot_time_cut"]))
    if merged:
        row.update(_rpc_content(c["source"], c["bx"], first))
    return row, all_idx, all_hits


def _cluster_hits(hits, eps, min_samples, settings, merged=False):
    """DBSCAN one (possibly merged) flat hit collection -> cluster records.

    ``hits`` is a :func:`_flat_hits` dict, or several of them run through
    :func:`_merge_hits`; ``merged`` adds the fields that only mean something
    when more than one system went in (the RPC timing estimate, the RPC hit
    content and which system the innermost hit came from).  A system with a
    rechit time (CSC ``Tpeak``) gets ``time`` / ``timeSpread`` /
    ``ootHitFrac`` from it.

    Events are processed as slices of the flat arrays; what is computed per
    cluster is in :func:`_one_cluster`.
    """
    counts = hits["counts"]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    has_time, has_truth = hits["has_time"], hits["has_truth"]

    fields = _cluster_fields(has_time, merged)
    out = {f: [] for f in fields}
    # Per cluster, every LLP clearing match_min_hits and its hit count.
    all_idx, all_hits = [], []
    nclu = np.zeros(len(counts), dtype=np.int64)

    for i in range(len(counts)):
        if counts[i] < min_samples:
            continue
        s = slice(offsets[i], offsets[i + 1])
        event = {name: hits[name][s] for name in _HIT_ARRAYS}
        labels = dbscan_labels(event["eta"], event["phi"], eps, min_samples)
        nclu[i] = labels.max() + 1
        for k in range(nclu[i]):
            in_cluster = labels == k
            cluster = {name: values[in_cluster] for name, values in event.items()}
            row, idx, n_hits = _one_cluster(cluster, has_truth, has_time, merged,
                                            hits["code"], settings)
            for f in fields:
                out[f].append(row[f])
            all_idx.append(idx)
            all_hits.append(n_hits)

    clusters = ak.zip({
        f: ak.unflatten(np.asarray(v, dtype=_FIELD_DTYPES.get(f, np.float64)), nclu)
        for f, v in out.items()})
    # (event, cluster, matched-LLP): variable length, so attached after the
    # zip of the per-cluster scalars rather than inside it.
    for name, vals in (("matchedLLPIdxAll", all_idx),
                       ("matchedLLPHitsAll", all_hits)):
        inner = ak.unflatten(
            np.concatenate(vals) if vals else np.zeros(0, dtype=np.int64),
            np.array([len(v) for v in vals], dtype=np.int64))
        clusters = ak.with_field(clusters, ak.unflatten(inner, nclu), name)
    return clusters


def _cluster_system(rechits, timefield, eps, min_samples, settings, system="csc"):
    """DBSCAN-cluster one rechit system -> jagged record array of clusters."""
    return _cluster_hits(_flat_hits(rechits, timefield, system), eps,
                         min_samples, settings)


def _cluster_merged(components, eps, min_samples, settings):
    """DBSCAN-cluster several rechit collections together.

    ``components`` is a list of ``(rechits, timefield, system)``; the first
    entry is the primary system, which names the resulting collection and
    supplies the ``time`` field.  Used by the RPC-merged configuration, where
    the RPC rechits of one region are clustered with the barrel (DT) or endcap
    (CSC) system they overlap instead of on their own, so a shower seen by both
    becomes one cluster with an RPC time attached.
    """
    hits = _merge_hits([_flat_hits(rechits, timefield, system)
                        for rechits, timefield, system in components])
    return _cluster_hits(hits, eps, min_samples, settings, merged=True)
