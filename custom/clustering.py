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

from .params import PARAMS

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
    # that is the field the historical ``time`` means -- CSC Tpeak.  RPC times
    # are reported separately by the rpcTime* fields.
    merged["has_time"] = components[0]["has_time"]
    merged["has_truth"] = all(c["has_truth"] for c in components)
    return merged


# The RPC fields a cluster carries, in both rpc_mode: merge and rpc_mode: match
# -- the same names and the same meanings, so one config set plots either.
RPC_CLUSTER_FIELDS = ("firstSystem", "nRPCHits", "nRPCHitsBx0", "rpcHitFrac",
                      "rpcTime", "rpcTimeMedian", "rpcTimeSpread",
                      "rpcTimeWeighted", "rpcTimeErr", "rpcTimeValidFrac",
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
            for f, v in _rpc_time(hits["time"][h][m] if n else empty,
                                  hits["timeError"][h][m] if n else empty,
                                  bx).items():
                out[f].append(v)

    ints = {"firstSystem", "nRPCHits", "nRPCHitsBx0"}
    for f in RPC_CLUSTER_FIELDS:
        values = np.asarray(out[f], dtype=np.int64 if f in ints else np.float64)
        clusters = ak.with_field(clusters, ak.unflatten(values, nclu), f)
    return clusters


def _rpc_time(times, errors, bx):
    """Timing of one cluster's RPC hits, from the rechit time and the BX.

    RPC is the only muon subdetector whose MDSNano rechits carry timing at
    all, so its hits are what date a merged cluster.  Two independent
    estimates, because the fine one is not always there:

    * the rechit time (ns), averaged plainly, by ``1/sigma^2`` from the
      per-hit ``TimeError``, and as a median (robust against the few badly
      mistimed hits a shower leaves).  A rechit whose ``TimeError`` is not
      positive has no valid time -- the whole MDSNano production line as of
      2026-08 stores ``Time = 0``, ``TimeError = -1`` -- and is left out, so
      these fields are NaN rather than a fake 0 when nothing was measured.
      ``rpcTimeValidFrac`` says how much of the cluster fed them.
    * the bunch crossing, which *is* filled: mean / median / RMS in BX units
      (multiply by 25 ns for a time), plus ``rpcOutOfTimeFrac``, the fraction
      of the cluster's RPC hits outside the in-time BX -- the coarse handle on
      a late, displaced shower.

    A cluster with no RPC hit at all gives NaN throughout.
    """
    fields = ("rpcTime", "rpcTimeMedian", "rpcTimeSpread", "rpcTimeWeighted",
              "rpcTimeErr", "rpcTimeValidFrac", "rpcBx", "rpcBxMedian",
              "rpcBxSpread", "rpcOutOfTimeFrac")
    if len(times) == 0:
        return dict.fromkeys(fields, np.nan)

    out = dict.fromkeys(fields, np.nan)
    valid = np.isfinite(times) & np.isfinite(errors) & (errors > 0)
    out["rpcTimeValidFrac"] = float(valid.mean())
    if valid.any():
        t, weights = times[valid], 1.0 / errors[valid] ** 2
        out["rpcTime"] = float(t.mean())
        out["rpcTimeMedian"] = float(np.median(t))
        out["rpcTimeSpread"] = float(t.std())
        out["rpcTimeWeighted"] = float((t * weights).sum() / weights.sum())
        out["rpcTimeErr"] = float(1.0 / np.sqrt(weights.sum()))

    good_bx = bx[np.isfinite(bx)]
    if len(good_bx):
        out["rpcBx"] = float(good_bx.mean())
        out["rpcBxMedian"] = float(np.median(good_bx))
        out["rpcBxSpread"] = float(good_bx.std())
        out["rpcOutOfTimeFrac"] = float((good_bx != 0).mean())
    return out


def _cluster_hits(hits, eps, min_samples, merged=False, params=None):
    """DBSCAN one (possibly merged) flat hit collection -> cluster records.

    ``hits`` is a :func:`_flat_hits` dict, or several of them run through
    :func:`_merge_hits`; ``merged`` adds the fields that only mean something
    when more than one system went in (the RPC timing estimate, the RPC hit
    content and which system the innermost hit came from).

    A system with a rechit time (CSC ``Tpeak``) gets ``time`` / ``timeSpread``
    / ``ootHitFrac`` from it -- the mean, the RMS and the fraction of the
    cluster's hits outside +-``oot_time_cut``.  The spread and the out-of-time
    fraction are the out-of-time-background handles: a real shower deposits
    every hit at one time, a cluster DBSCAN assembled from unrelated pile-up
    and cavern hits does not.
    """
    from sklearn.cluster import DBSCAN

    p = PARAMS if params is None else params
    match_min_hits, oot_cut = p["match_min_hits"], p["oot_time_cut"]
    counts = hits["counts"]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    eta, phi = hits["eta"], hits["phi"]
    xs, ys, zs = hits["x"], hits["y"], hits["z"]
    station, dist = hits["station"], hits["dist"]
    layerkey, zlayerkey, chamber = hits["layerkey"], hits["zlayerkey"], hits["chamber"]
    llpidx, source = hits["llpidx"], hits["source"]
    has_truth = hits["has_truth"]
    tvals = hits["time"] if hits["has_time"] else None
    primary_code, rpc_code = hits["code"], SOURCE_CODES["rpc"]

    fields = ["size", "eta", "phi", "x", "y", "z", "r", "etaSpread", "phiSpread",
              "rSpread", "zSpread",
              "nStation", "stationSpan", "avgStation", "maxStationFrac",
              "nLayer", "layerHitsMean", "layerHitsRMS", "layerHitsRelRMS",
              "maxLayerFrac",
              "nZLayer", "zLayerHitsMean", "zLayerHitsRMS", "zLayerHitsRelRMS",
              "maxZLayerFrac",
              "firstChamber", "firstStation",
              "nMatchedHits", "matchedLLPIdx", "matched", "nMatchedLLP", "hasTruth"]
    if tvals is not None:
        fields += ["time", "timeSpread", "ootHitFrac"]
    if merged:
        fields += ["firstSystem", "nRPCHits", "nRPCHitsBx0", "rpcHitFrac",
                   "rpcTime", "rpcTimeMedian", "rpcTimeSpread",
                   "rpcTimeWeighted", "rpcTimeErr", "rpcTimeValidFrac",
                   "rpcBx", "rpcBxMedian", "rpcBxSpread", "rpcOutOfTimeFrac"]
    out = {f: [] for f in fields}
    nclu = np.zeros(len(counts), dtype=np.int64)

    for i in range(len(counts)):
        n = counts[i]
        if n < min_samples:
            continue
        s = slice(offsets[i], offsets[i + 1])
        e, p = eta[s], phi[s]
        dphi = p[:, None] - p[None, :]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dmat = np.sqrt((e[:, None] - e[None, :]) ** 2 + dphi ** 2)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="precomputed").fit_predict(dmat)
        for k in range(labels.max() + 1):
            m = labels == k
            cphi = np.arctan2(np.sin(p[m]).sum(), np.cos(p[m]).sum())
            dphic = (p[m] - cphi + np.pi) % (2 * np.pi) - np.pi
            idx = llpidx[s][m]
            idx = idx[idx >= 0]
            if len(idx):
                uniq, cnt = np.unique(idx, return_counts=True)
                best, nbest = int(uniq[cnt.argmax()]), int(cnt.max())
                # How many distinct LLPs clear the match threshold in this
                # cluster (a cluster can overlap more than one LLP's shower).
                nmatched_llp = int((cnt >= match_min_hits).sum())
            else:
                best, nbest, nmatched_llp = -1, 0, 0
            matched = nbest >= match_min_hits
            out["size"].append(int(m.sum()))
            out["eta"].append(float(e[m].mean()))
            out["phi"].append(float(cphi))
            out["x"].append(float(xs[s][m].mean()))
            out["y"].append(float(ys[s][m].mean()))
            out["z"].append(float(zs[s][m].mean()))
            out["r"].append(float(np.hypot(xs[s][m].mean(), ys[s][m].mean())))
            out["etaSpread"].append(float(e[m].std()))
            out["phiSpread"].append(float(np.sqrt(np.mean(dphic ** 2))))
            out["rSpread"].append(float(np.hypot(xs[s][m], ys[s][m]).std()))
            out["zSpread"].append(float(zs[s][m].std()))
            # Hit multiplicity across stations: a shower is contained in one
            # station, a punch-through muon crosses several.
            st = np.abs(station[s][m])
            _, scnt = np.unique(st, return_counts=True)
            out["nStation"].append(len(scnt))
            out["stationSpan"].append(int(st.max() - st.min()))
            out["avgStation"].append(float(st.mean()))
            out["maxStationFrac"].append(float(scnt.max() / scnt.sum()))
            # Hit multiplicity across layers: a muon leaves ~1 hit per layer
            # (uniform), a shower a strongly varying number ("fuzziness").
            _, lcnt = np.unique(layerkey[s][m], return_counts=True)
            lmean = float(lcnt.mean())
            out["nLayer"].append(len(lcnt))
            out["layerHitsMean"].append(lmean)
            out["layerHitsRMS"].append(float(lcnt.std()))
            out["layerHitsRelRMS"].append(float(lcnt.std()) / lmean)
            out["maxLayerFrac"].append(float(lcnt.max() / lcnt.sum()))
            # Same moments with the layer plane taken from quantized global z
            # (CSC only in practice; see _flat_zlayer_key for the caveat on
            # ntuples that store one z per chamber).
            _, zcnt = np.unique(zlayerkey[s][m], return_counts=True)
            zmean = float(zcnt.mean())
            out["nZLayer"].append(len(zcnt))
            out["zLayerHitsMean"].append(zmean)
            out["zLayerHitsRMS"].append(float(zcnt.std()))
            out["zLayerHitsRelRMS"].append(float(zcnt.std()) / zmean)
            out["maxZLayerFrac"].append(float(zcnt.max() / zcnt.sum()))
            # Where the cluster starts: the hit closest to the interaction
            # point.  A punch-through begins at an innermost chamber.
            first = int(np.argmin(dist[s][m]))
            out["firstChamber"].append(int(chamber[s][m][first]))
            out["firstStation"].append(int(abs(station[s][m][first])))
            out["nMatchedHits"].append(nbest)
            out["matchedLLPIdx"].append(best if matched else -1)
            out["matched"].append(matched)
            out["nMatchedLLP"].append(nmatched_llp)
            out["hasTruth"].append(has_truth)
            src = source[s][m] if merged else None
            if tvals is not None:
                # ``time`` is the primary system's time only (CSC Tpeak): the
                # RPC hits a merged cluster picked up are reported apart, by
                # the rpcTime* fields, and are on a different clock.
                t = tvals[s][m] if src is None else tvals[s][m][src == primary_code]
                t = t[np.isfinite(t)]
                # The spread separates a real shower (every hit from the same
                # particle, so one time) from a cluster DBSCAN built out of
                # unrelated in-time and out-of-time hits; ootHitFrac says how
                # much of it sits outside the in-time window.
                out["time"].append(float(t.mean()) if len(t) else np.nan)
                out["timeSpread"].append(float(t.std()) if len(t) else np.nan)
                out["ootHitFrac"].append(float((np.abs(t) > oot_cut).mean())
                                         if len(t) else np.nan)
            if merged:
                isrpc = src == rpc_code
                rpc_bx = hits["bx"][s][m][isrpc]
                out["firstSystem"].append(int(src[first]))
                out["nRPCHits"].append(int(isrpc.sum()))
                out["nRPCHitsBx0"].append(int(np.count_nonzero(rpc_bx == 0)))
                out["rpcHitFrac"].append(float(isrpc.sum() / m.sum()))
                for f, v in _rpc_time(hits["time"][s][m][isrpc],
                                      hits["timeError"][s][m][isrpc],
                                      rpc_bx).items():
                    out[f].append(v)
        nclu[i] = labels.max() + 1

    dtypes = {"size": np.int64, "nStation": np.int64, "stationSpan": np.int64,
              "nLayer": np.int64, "nZLayer": np.int64, "nMatchedHits": np.int64,
              "firstChamber": np.int64, "firstStation": np.int64,
              "matchedLLPIdx": np.int64, "matched": np.bool_,
              "nMatchedLLP": np.int64, "hasTruth": np.bool_,
              "firstSystem": np.int64, "nRPCHits": np.int64,
              "nRPCHitsBx0": np.int64}
    return ak.zip({f: ak.unflatten(np.asarray(v, dtype=dtypes.get(f, np.float64)), nclu)
                   for f, v in out.items()})


def _cluster_system(rechits, timefield, eps, min_samples, system="csc",
                    params=None):
    """DBSCAN-cluster one rechit system -> jagged record array of clusters.

    *params* are the resolved settings (:data:`PARAMS` when omitted); the
    pipeline always passes them explicitly, so nothing below derive() reads
    the module-level settings.
    """
    return _cluster_hits(_flat_hits(rechits, timefield, system), eps,
                         min_samples, params=params)


def _cluster_merged(components, eps, min_samples, params=None):
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
    return _cluster_hits(hits, eps, min_samples, merged=True, params=params)
