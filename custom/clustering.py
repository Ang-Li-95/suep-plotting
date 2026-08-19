"""DBSCAN clustering of the muon-system rechits.

:func:`_cluster_system` turns one rechit collection into a jagged record array
of clusters (dR metric in eta-phi, per-cluster shape/position/truth-match
fields).  The ``_flat_*_key`` helpers build the flat integer ids it counts hits
over: the depth segment ("layer"), the chamber-type code, and the layer plane
read off the quantized global z.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .params import PARAMS


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
    so the code is built from the station and the ring-like coordinate — the
    wheel for DT, the ring for RPC — in the same ``station * 10 + ring`` layout.
    RPC additionally offsets the endcap by 100, since a barrel and an endcap
    chamber can otherwise share a code.
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
    2.2 cm in ME1/1; 132 planes in total) — finer than the chamber unit of
    :func:`_flat_layer_key`.  CAVEAT: the central MDSNano v2 background
    production stores ONE z value per chamber (in-chamber layer information
    is dropped), so the ``zLayer*`` cluster fields are only comparable
    between samples whose ntuples keep the true per-layer z (the private
    signal production does; a reprocessed background sample would).  Only
    meaningful for CSC — for barrel systems z runs along the wires, not
    through the layers.
    """
    return np.round(
        np.asarray(ak.flatten(rechits.Z), dtype=np.float64) * 10.0
    ).astype(np.int64)


def _cluster_system(rechits, timefield, eps, min_samples):
    """DBSCAN-cluster one rechit system -> jagged record array of clusters.

    The rechit branches are materialized to flat numpy once per chunk and
    events are processed as offset slices (per-event awkward indexing is slow).
    """
    from sklearn.cluster import DBSCAN

    match_min_hits = PARAMS["match_min_hits"]
    counts = np.asarray(ak.num(rechits.Eta))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    eta = np.asarray(ak.flatten(rechits.Eta), dtype=np.float64)
    phi = np.asarray(ak.flatten(rechits.Phi), dtype=np.float64)
    xs = np.asarray(ak.flatten(rechits.X), dtype=np.float64)
    ys = np.asarray(ak.flatten(rechits.Y), dtype=np.float64)
    zs = np.asarray(ak.flatten(rechits.Z), dtype=np.float64)
    station = np.asarray(ak.flatten(rechits.Station))
    layerkey = _flat_layer_key(rechits)
    zlayerkey = _flat_zlayer_key(rechits)
    chamber = _flat_chamber_code(rechits)
    # Distance to the nominal interaction point, to pick a cluster's first hit.
    dist = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    has_truth = "llpIdx" in rechits.fields
    llpidx = (np.asarray(ak.flatten(rechits.llpIdx)) if has_truth
              else np.full(len(eta), -1, dtype=np.int64))
    tvals = (np.asarray(ak.flatten(rechits[timefield]), dtype=np.float64)
             if timefield is not None else None)

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
        fields.append("time")
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
            if tvals is not None:
                out["time"].append(float(tvals[s][m].mean()))
        nclu[i] = labels.max() + 1

    dtypes = {"size": np.int64, "nStation": np.int64, "stationSpan": np.int64,
              "nLayer": np.int64, "nZLayer": np.int64, "nMatchedHits": np.int64,
              "firstChamber": np.int64, "firstStation": np.int64,
              "matchedLLPIdx": np.int64, "matched": np.bool_,
              "nMatchedLLP": np.int64, "hasTruth": np.bool_}
    return ak.zip({f: ak.unflatten(np.asarray(v, dtype=dtypes.get(f, np.float64)), nclu)
                   for f, v in out.items()})
