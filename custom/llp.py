"""The gen-level ``events.llp`` collection and its rechit shape variables.

:func:`_build_llps` reads the LLPs out of the unpruned ``SUEPGenPart`` table
(kinematics, decay vertex, opening angle, decay-volume flags) and
:func:`_empty_llps` provides the same fields, empty, for samples without truth
branches.  :func:`_llp_rechit_dr` (with its compiled :func:`_dr_kernel`)
measures the eta-phi spread of the rechits truth-matched to each LLP.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from numba import njit

from .params import CSC_RMAX, CSC_ZMAX, CSC_ZMIN, DT_RMAX, DT_RMIN, DT_ZMAX
from .params import LLP_PDGID, PARAMS, RPC_EC_RMAX, RPC_EC_ZMAX, RPC_EC_ZMIN, STEPS


def _llpidx_is_genpart_index(events):
    """Detect whether rechit ``llpIdx`` indexes SUEPGenPart directly.

    Post-Geant4-fix files store the SUEPGenPart index (``SUEPGenPart.pdgId`` at
    that index is the LLP, 999999); pre-fix files store the ordinal LLP index
    (which points at low-index beam/hard-process particles instead).  Both
    conventions keep ``llpIdx < nSUEPGenPart``, so indexing is always safe.
    Returns True (genpart-index convention) when no matched hits are present.

    NOT called at runtime: the convention is taken from the
    ``llpidx_convention`` parameter (default "genpart").  Kept as a standalone
    check for validating that setting on a new production.
    """
    idx = events.cscRechits.llpIdx
    sel = idx[idx >= 0]
    matched_pdg = ak.flatten(events.SUEPGenPart.pdgId[sel])
    if len(matched_pdg) == 0:
        return True
    return bool(ak.mean(matched_pdg == LLP_PDGID) > 0.5)


def _build_llps(events):
    """LLP collection from the unpruned SUEPGenPart table.

    The decay vertex is the production vertex of the LLP's daughters
    (every LLP in these samples has exactly two daughters stored).
    """
    gp = events.SUEPGenPart
    gp_idx = ak.local_index(gp.pdgId, axis=1)
    is_llp = gp.pdgId == LLP_PDGID
    gidx = gp_idx[is_llp]              # SUEPGenPart index of each LLP
    lidx = ak.local_index(gidx, axis=1)  # ordinal 0..nLLP-1 within the event

    mom = gp.genPartIdxMother
    has_mom = mom >= 0
    mom_pdg = gp.pdgId[ak.where(has_mom, mom, 0)]
    is_dau = has_mom & (mom_pdg == LLP_PDGID)
    dau_mom = mom[is_dau]

    # first daughter per LLP: (nLLP x nDau) comparison of int indices only
    match = gidx[:, :, None] == dau_mom[:, None, :]
    has_dau = ak.any(match, axis=2)
    first = ak.fill_none(ak.argmax(match, axis=2), 0)
    dvx = ak.where(has_dau, gp.vx[is_dau][first], np.nan)
    dvy = ak.where(has_dau, gp.vy[is_dau][first], np.nan)
    dvz = ak.where(has_dau, gp.vz[is_dau][first], np.nan)

    # Opening angle between the two decay daughters (every LLP has exactly two).
    # Boost-driven: a more boosted LLP decays into a tighter pair, which is what
    # ultimately sets the eta-phi spread of its rechits.  Second daughter = the
    # matched one that is not `first`; NaN if the LLP has fewer than two stored.
    dau_local = ak.local_index(dau_mom, axis=1)
    match2 = match & (dau_local[:, None, :] != first[:, :, None])
    has_two = ak.sum(match, axis=2) >= 2
    second = ak.fill_none(ak.argmax(match2, axis=2), 0)
    d_eta, d_phi = gp.eta[is_dau], gp.phi[is_dau]
    t1 = 2 * np.arctan(np.exp(-d_eta[first]))
    t2 = 2 * np.arctan(np.exp(-d_eta[second]))
    dphi_d = d_phi[first] - d_phi[second]
    cos_open = (np.sin(t1) * np.sin(t2) * np.cos(dphi_d)
                + np.cos(t1) * np.cos(t2))
    # np.clip is not a ufunc, so it can't dispatch over the jagged array; clamp
    # with the min/max ufuncs before arccos (guards float round-off past +-1).
    cos_open = np.minimum(np.maximum(cos_open, -1.0), 1.0)
    opening_angle = ak.where(has_two, np.arccos(cos_open), np.nan)

    pt, eta, mass = gp.pt[is_llp], gp.eta[is_llp], gp.mass[is_llp]
    decay_r = np.hypot(dvx, dvy)
    abs_z = abs(dvz)

    in_csc = (abs_z > CSC_ZMIN) & (abs_z < CSC_ZMAX) & (decay_r < CSC_RMAX)
    in_dt = (decay_r > DT_RMIN) & (decay_r < DT_RMAX) & (abs_z < DT_ZMAX)
    in_rpc = in_dt | ((abs_z > RPC_EC_ZMIN) & (abs_z < RPC_EC_ZMAX)
                      & (decay_r < RPC_EC_RMAX))

    # Proper decay length: L_proper = L_lab / (beta*gamma) = L_lab * m / p,
    # where L_lab is the 3D distance from the LLP production vertex to its
    # decay vertex.  Distributed exponentially with mean = nominal ctau.
    p = pt * np.cosh(eta)
    l_lab = np.sqrt((dvx - gp.vx[is_llp]) ** 2 + (dvy - gp.vy[is_llp]) ** 2
                    + (dvz - gp.vz[is_llp]) ** 2)
    betagamma = p / mass

    return ak.zip({
        "pt": pt,
        "eta": eta,
        "phi": gp.phi[is_llp],
        "mass": mass,
        "energy": np.hypot(p, mass),
        "gidx": gidx,
        "lidx": lidx,
        "decayR": decay_r,
        "decayZ": dvz,
        "Llab": l_lab,
        "betagamma": betagamma,
        "ctau": l_lab / betagamma,
        "openingAngle": opening_angle,
        "inCSC": in_csc,
        "inDT": in_dt,
        "inRPC": in_rpc,
    })


def _empty_llps(events):
    """Zero-length llp collection (background samples without SUEPGenPart).

    Same fields as the real collection — for the enabled steps only, so the
    presence of a field means the same thing on signal and background.
    """
    counts = np.zeros(len(events), dtype=np.int64)

    def empty(dtype):
        return ak.unflatten(np.zeros(0, dtype=dtype), counts)

    f64 = ["pt", "eta", "phi", "mass", "energy", "decayR", "decayZ",
           "Llab", "betagamma", "ctau", "openingAngle"]
    i64 = ["gidx", "lidx"]
    boo = ["inCSC", "inDT", "inRPC"]
    if "llp_hits" in STEPS:
        i64 += ["nHitsCSC", "nHitsDT", "nHitsRPC", "nHitsTotal"]
    if "llp_reco" in STEPS:
        f64 += ["clusterHitFracCSC", "clusterHitFracDT", "clusterHitFracRPC"]
        i64 += ["nRecoClusterCSC", "nRecoClusterDT", "nRecoClusterRPC"]
        boo += ["recoCSC", "recoDT", "recoRPC", "reco"]
    if "llp_shape" in STEPS:
        f64 += [f + sys for sys in ("CSC", "DT", "RPC", "Total")
                for f in _dr_field_names()]
    fields = {f: empty(np.float64) for f in f64}
    fields.update({f: empty(np.int64) for f in i64})
    fields.update({f: empty(np.bool_) for f in boo})
    return ak.zip(fields)


def _dr_field_names():
    """Names of the per-LLP rechit-spread fields (without the system suffix)."""
    return ("drPairMin", "drPairMax", "drMax") + tuple(
        f"dr{int(round(q * 100))}" for q in PARAMS["dr_quantiles"])


def _llp_rechit_dr(events, coll_names, keys, hit_dr=None):
    """Eta-phi spread of the rechits truth-matched to each LLP.

    Pools the rechit collections named in ``coll_names`` (one system, or all
    three for the combined shape) and, for every LLP with at least one matched
    rechit, computes

    ``drPairMin`` / ``drPairMax``
        smallest / largest dR between any two of the LLP's rechits (closest
        pair and shower "diameter").
    ``dr50`` / ``dr80`` / ``dr90`` / ``drMax``
        radius around the rechit centroid containing 50 / 80 / 90 / 100 % of
        them, i.e. the cone size needed to collect that fraction of the hits.

    LLPs with no matched rechit get NaN in every field (silently dropped at
    fill time); ``drPairMin/Max`` are NaN as well for a single-hit LLP.

    ``keys`` is the per-event list of LLP identifiers the rechit ``llpIdx``
    branches are compared against (``llp.gidx`` or ``llp.lidx``).  ``hit_dr``,
    if given, is a flat per-rechit array (single collection only) filled in
    place with each matched rechit's dR to its LLP's centroid.

    Returns a dict of flat per-LLP arrays, to be unflattened with ``ak.num(keys)``.
    """
    nllp = np.asarray(ak.num(keys))
    keys_flat = np.asarray(ak.flatten(keys), dtype=np.int64)
    llp_off = np.concatenate([[0], np.cumsum(nllp)]).astype(np.int64)

    n_llp = len(keys_flat)
    pair_min = np.full(n_llp, np.nan)
    pair_max = np.full(n_llp, np.nan)
    dr_max = np.full(n_llp, np.nan)
    quantiles = PARAMS["dr_quantiles"]
    dr_q = np.full((len(quantiles), n_llp), np.nan)

    def result():
        out = {"drPairMin": pair_min, "drPairMax": pair_max, "drMax": dr_max}
        for t, q in enumerate(quantiles):
            out[f"dr{int(round(q * 100))}"] = dr_q[t]
        return out

    # Materialize the matched rechits of every requested collection to flat
    # numpy once (per-event awkward indexing is slow), as in _cluster_system.
    evts, etas, phis, idxs, srcs = [], [], [], [], []
    for name in coll_names:
        rechits = events[name]
        if "llpIdx" not in rechits.fields:
            continue
        counts = np.asarray(ak.num(rechits.Eta))
        llpidx = np.asarray(ak.flatten(rechits.llpIdx), dtype=np.int64)
        m = llpidx >= 0
        evts.append(np.repeat(np.arange(len(counts), dtype=np.int64), counts)[m])
        etas.append(np.asarray(ak.flatten(rechits.Eta), dtype=np.float64)[m])
        phis.append(np.asarray(ak.flatten(rechits.Phi), dtype=np.float64)[m])
        idxs.append(llpidx[m])
        srcs.append(np.flatnonzero(m).astype(np.int64))
    if not evts or sum(len(e) for e in evts) == 0:
        return result()

    # One stable sort by event groups the pooled hits the way the per-event
    # loop used to: all CSC hits of the event, then DT, then RPC.
    evt = np.concatenate(evts)
    grp = np.argsort(evt, kind="stable")
    hit_eta = np.concatenate(etas)[grp]
    hit_phi = np.concatenate(phis)[grp]
    hit_idx = np.concatenate(idxs)[grp]
    hit_src = (np.concatenate(srcs)[grp] if hit_dr is not None
               else np.empty(0, dtype=np.int64))
    ev_off = np.searchsorted(evt[grp], np.arange(len(nllp) + 1)).astype(np.int64)

    _dr_kernel(ev_off, hit_eta, hit_phi, hit_idx, hit_src, keys_flat, llp_off,
               np.asarray(quantiles, dtype=np.float64),
               pair_min, pair_max, dr_max, dr_q,
               hit_dr if hit_dr is not None else np.empty(0),
               hit_dr is not None, int(PARAMS["pair_max_hits"]))
    return result()


@njit(cache=True)
def _dr_kernel(ev_off, hit_eta, hit_phi, hit_idx, hit_src, keys_flat, llp_off,
               quantiles, pair_min, pair_max, dr_max, dr_q, hit_dr, do_hits,
               pair_max_hits):
    """Compiled inner loop of :func:`_llp_rechit_dr` (see it for definitions).

    Every array is flat and preallocated by the caller; the five output arrays
    are written in place.  ``ev_off`` slices the matched-hit arrays per event,
    ``llp_off`` slices ``keys_flat`` and the output rows per event.
    ``pair_max_hits`` is passed in rather than read from :data:`PARAMS`, which
    numba would freeze into the cached machine code at first compile.

    Compiled because the groups are tiny (~3 rechits per LLP): in numpy this
    loop was ~17x slower, spending almost all of it on per-call dispatch rather
    than arithmetic.  Results match the numpy version to the last ulp (numpy
    sums pairwise, this sums in a plain loop).
    """
    two_pi = 2.0 * np.pi
    for i in range(len(ev_off) - 1):
        a0, a1 = ev_off[i], ev_off[i + 1]
        k0, k1 = llp_off[i], llp_off[i + 1]
        if a1 <= a0 or k1 <= k0:
            continue
        n = a1 - a0
        # Group this event's hits by llpIdx with one sort, instead of one mask
        # per LLP (events hold up to O(50) LLPs, most without any rechit).
        order = np.argsort(hit_idx[a0:a1], kind="mergesort")

        a = 0
        while a < n:
            key = hit_idx[a0 + order[a]]
            b = a + 1
            while b < n and hit_idx[a0 + order[b]] == key:
                b += 1
            j = -1
            for t in range(k0, k1):
                if keys_flat[t] == key:
                    j = t
                    break
            if j < 0:              # hits from a gen particle that is not an LLP
                a = b
                continue

            m = b - a
            e = np.empty(m, dtype=np.float64)
            p = np.empty(m, dtype=np.float64)
            for t in range(m):
                g = a0 + order[a + t]
                e[t] = hit_eta[g]
                p[t] = hit_phi[g]

            # Centroid: circular mean in phi, so hits either side of +-pi
            # average correctly (same convention as the DBSCAN clusters).
            sin_sum = 0.0
            cos_sum = 0.0
            eta_sum = 0.0
            for t in range(m):
                sin_sum += np.sin(p[t])
                cos_sum += np.cos(p[t])
                eta_sum += e[t]
            cphi = np.arctan2(sin_sum, cos_sum)
            emean = eta_sum / m

            dr = np.empty(m, dtype=np.float64)
            biggest = 0.0
            for t in range(m):
                dphi = (p[t] - cphi + np.pi) % two_pi - np.pi
                dr[t] = np.hypot(e[t] - emean, dphi)
                if dr[t] > biggest:
                    biggest = dr[t]
            dr_max[j] = biggest
            quants = np.quantile(dr, quantiles)
            for t in range(len(quantiles)):
                dr_q[t, j] = quants[t]
            if do_hits:
                for t in range(m):
                    hit_dr[hit_src[a0 + order[a + t]]] = dr[t]

            if m >= 2:
                step = 1 + (m - 1) // pair_max_hits
                lo = np.inf
                hi = -np.inf
                for u in range(0, m, step):
                    for v in range(u + step, m, step):
                        dphi = (p[u] - p[v] + np.pi) % two_pi - np.pi
                        d = np.hypot(e[u] - e[v], dphi)
                        if d < lo:
                            lo = d
                        if d > hi:
                            hi = d
                if hi >= 0.0:
                    pair_min[j] = lo
                    pair_max[j] = hi
            a = b
