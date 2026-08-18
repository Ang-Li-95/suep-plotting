"""Jet energy corrections (JEC) and resolution smearing (JER).

JEC/JER rescale the jet four-momentum, so — unlike the weights in
``corrections.yaml`` / ``reweights.yaml`` — they must run *before* selections
and histogram fills.  Call :func:`correct_jets` from ``custom/columns.py``::

    from suep_plot.jme import correct_jets

    def derive(events):
        events = correct_jets(events)                    # 2024 Summer24 MC
        return events

Every expression that touches ``events.Jet`` (HT, jet pT spectra, selections)
then automatically uses corrected, smeared jets.  For systematics::

    events = correct_jets(events, variation="jec_up")    # jec_down / jer_up / jer_down

The **JEC** is coffea's own :class:`~coffea.jetmet_tools.CorrectedJetsFactory`,
fed the jsonpog payloads through coffea's ``correctionlib_adapters``.  The
**JER smearing** is the official ``JERSmear`` correctionlib payload shipped next
to this module, which owns the hybrid-vs-stochastic choice and the deterministic
seeding.  This module is the wiring around the two:

1. resolve the payload tags for the era (``jet_jerc.json.gz``), including the
   ``*_DATA`` tag whose compound is run-dependent;
2. build the per-jet columns the factory reads through its ``name_map``:
   ``pt_raw``/``mass_raw`` (undoing ``rawFactor``), ``event_rho``, ``run``, and
   ``pt_gen`` -- the gen pT of jets matched within ``dR < 0.2`` and ``3 sigma``,
   which is what selects the hybrid smearing method per jet;
3. pick the requested variation out of the factory's output
   (``JES_Total.up/down``, ``JER.up/down``);
4. rebuild the Type-1 MET from the **raw** MET (see :func:`_rebuild_met`):
   subtract the vector sum of ``pT_L2L3Res - pT_L1`` over both jet collections
   (``Jet`` and ``CorrT1METJet``), muon-subtracted, for jets with corrected
   pT > 15 GeV, ``|eta| < 5.2`` and EM fraction < 0.9;
5. re-sort jets by the new pT and drop the helper columns.

``pt`` and ``mass`` (and the MET's ``pt``/``phi``) are replaced in place.

Payloads resolve via :func:`payload_path`: ``$CORRECTIONLIB_DATA`` first, then
the era's CAT campaign directory (:data:`CAT_METADATA`), then cvmfs
jsonpog-integration.  The campaign directory matters -- jsonpog's
``2024_Summer24`` still carries only ``Summer24Prompt24_V1``, while the 2024
recommendation is ``V5``.
"""

from __future__ import annotations

import functools
import pathlib

import awkward as ak
import numpy as np

from .corrections import _resolve_file

# JME ships Summer23BPix JER payloads inside the 2024_Summer24 file until the
# Summer24 JER is released; these defaults match the file contents.
# CMS Common Analysis Tools publishes the per-campaign JME payloads here, ahead
# of (and sometimes instead of) the jsonpog-integration copies: jsonpog's
# 2024_Summer24 still only carries Summer24Prompt24_V1.
CAT_METADATA = "/cvmfs/cms-griddata.cern.ch/cat/metadata"

DEFAULTS = {
    "2024_Summer24": {
        # The campaign directory the payloads come from.  ``changes.md`` next to
        # it records why V5 exists: V3->V4 bumped the tag without actually
        # updating the L2L3Residual payloads, V5 is the fixed one.
        "payload_dir": (f"{CAT_METADATA}/JME/"
                        "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest"),
        "jec_tag": "Summer24Prompt24_V5_MC",
        # Data gets the residual-corrected DATA tag; Summer24Prompt24 ships one
        # payload for the whole year, so no run-range mapping is needed.
        "jec_tag_data": "Summer24Prompt24_V5_DATA",
        "jer_tag": "Summer24Prompt24_JRV2_MC",
    },
}

# The smearing payload is era-independent and lives in its own CAT directory.
SMEAR_DIRS = (f"{CAT_METADATA}/JME/JER-Smearing/latest",)

_VARIATIONS = ("nominal", "jec_up", "jec_down", "jer_up", "jer_down")


def payload_path(era: str, filename: str) -> str:
    """Locate a JME payload for *era*.

    Order: ``$CORRECTIONLIB_DATA`` (via the ``auto:`` lookup, so a local copy
    always wins), then the era's CAT campaign directory, then jsonpog.  Shared
    with ``custom/columns.py`` so the jet ID and the JEC come from one campaign.
    """
    import os

    override = os.path.join(os.environ.get("CORRECTIONLIB_DATA", ""),
                            "JME", era, filename)
    if os.environ.get("CORRECTIONLIB_DATA") and os.path.isfile(override):
        return override

    campaign = DEFAULTS.get(era, {}).get("payload_dir")
    if campaign:
        candidate = os.path.join(campaign, filename)
        if os.path.isfile(candidate):
            return candidate

    return _resolve_file(f"auto:JME/{era}/{filename}")


@functools.lru_cache(maxsize=None)
def _smear_cset():
    """The ``JERSmear`` payload, shipped next to this module.

    jsonpog does not distribute ``jer_smear.json.gz`` per era -- it is one tiny
    era-independent file -- so it lives in the package rather than being looked
    up on cvmfs.  ``$CORRECTIONLIB_DATA`` / cvmfs still win if they have one.
    """
    import correctionlib

    for directory in SMEAR_DIRS:
        candidate = pathlib.Path(directory) / "jer_smear.json.gz"
        if candidate.is_file():
            return correctionlib.CorrectionSet.from_file(str(candidate))
    try:
        return correctionlib.CorrectionSet.from_file(
            _resolve_file("auto:JME/jer_smear.json.gz"))
    except FileNotFoundError:
        local = pathlib.Path(__file__).with_name("jer_smear.json.gz")
        if not local.is_file():
            raise
        return correctionlib.CorrectionSet.from_file(str(local))


@functools.lru_cache(maxsize=None)
def _load_cset(path: str):
    import correctionlib
    return correctionlib.CorrectionSet.from_file(_resolve_file(path))


def _evaluate(corr, available: dict) -> np.ndarray:
    """Evaluate a correction, matching its declared input names to arrays."""
    args = []
    for inp in corr.inputs:
        if inp.name not in available:
            raise KeyError(f"correction '{corr.name}' needs input '{inp.name}' "
                           f"(have: {list(available)})")
        args.append(available[inp.name])
    return corr.evaluate(*args)


def _matched_gen_pt(jets, genjets, pt_corr_flat, resolution_flat, counts):
    """Gen-jet pT per jet for JER scaling; -1 where the jet is unmatched.

    Matching = valid ``genJetIdx`` + ``dR < 0.2`` (half the AK4 cone) +
    ``|pt - pt_gen| < 3 sigma_JER * pt`` per the JME prescription.
    """
    idx = jets.genJetIdx
    valid = (idx >= 0) & (idx < ak.num(genjets))
    gen_padded = ak.pad_none(genjets, 1)          # safe to index in empty events
    matched = gen_padded[ak.where(valid, idx, 0)]

    gen_pt = ak.fill_none(ak.where(valid, matched.pt, -1.0), -1.0)
    deta = ak.fill_none(jets.eta - matched.eta, 99.0)
    dphi = ak.fill_none((jets.phi - matched.phi + np.pi) % (2 * np.pi) - np.pi, 99.0)
    dr_ok = ak.fill_none((deta**2 + dphi**2) < 0.2**2, False) & valid

    gen_pt_flat = np.asarray(ak.flatten(gen_pt), dtype=np.float64)
    dr_ok_flat = np.asarray(ak.flatten(dr_ok))
    close = np.abs(pt_corr_flat - gen_pt_flat) < 3.0 * resolution_flat * pt_corr_flat
    return np.where(dr_ok_flat & close, gen_pt_flat, -1.0)


# NanoAOD's raw counterpart for each MET flavour: the Type-1 correction is
# rebuilt from the raw MET, never from the already-corrected one.
RAW_MET = {"PuppiMET": "RawPuppiMET", "PFMET": "RawPFMET", "MET": "RawMET"}

# Type-1 jet selection (JME prescription).
T1_MIN_PT = 15.0
T1_MAX_ABSETA = 5.2
T1_MAX_EMEF = 0.9


def _t1_delta(area, eta, phi, pt_raw, musub, emef, cset, jec_tag, algo, rho,
              run=None, scale=None, dphi=None):
    """Per-jet Type-1 momentum change, as (dx, dy) flat arrays.

    The correction propagated into MET is ``pT_L2L3Res - pT_L1``: the L1
    (pileup-offset) part is what MET already accounts for, so only what sits
    *above* L1 may be moved.  Both are evaluated on the muon-subtracted raw pT,
    because the muon momentum belongs to the muon, not to the jet, as far as MET
    is concerned: MET counts the muon directly, so propagating a correction
    derived over the muon's momentum too would double count it.  The jets handed
    back to the caller are *not* muon-subtracted -- that asymmetry is the
    prescription, not an oversight.  *scale* optionally carries the extra factor
    the jets themselves received (JER smearing, a JES variation).

    Removing the muon also moves the jet's direction, so the projection uses
    ``phi + muonSubtrDeltaPhi`` (*dphi*) -- the axis of what is left after the
    subtraction, which is the object whose momentum change MET must absorb.
    """
    ptmu = pt_raw * (1.0 - musub)
    available = {"JetA": area, "JetEta": eta, "JetPt": ptmu, "Rho": rho,
                 "JetPhi": phi}
    if run is not None:
        available["run"] = run

    pt_l1 = ptmu * _evaluate(cset[f"{jec_tag}_L1FastJet_{algo}"], available)
    pt_corr = ptmu * _evaluate(cset.compound[f"{jec_tag}_L1L2L3Res_{algo}"],
                               available)
    if scale is not None:
        pt_corr = pt_corr * scale

    use = ((pt_corr > T1_MIN_PT) & (np.abs(eta) < T1_MAX_ABSETA)
           & (emef < T1_MAX_EMEF))
    dpt = np.where(use, pt_corr - pt_l1, 0.0)

    axis = phi if dphi is None else phi + dphi
    return dpt * np.cos(axis), dpt * np.sin(axis)


def _musub_dphi(collection, flat):
    """``muonSubtrDeltaPhi`` for *collection*, or zeros if it is not stored."""
    if "muonSubtrDeltaPhi" in collection.fields:
        return flat(collection.muonSubtrDeltaPhi)
    return None


def _rebuild_met(events, met_name, cset, jec_tag, algo, scale, counts):
    """Replace *met_name* with a Type-1 correction rebuilt from the raw MET.

    Two jet collections enter, and both are needed for the sum to be complete:
    ``Jet`` (raw pT >= 15) and ``CorrT1METJet``, the low-pT jets NanoAOD stores
    with only the fields a Type-1 rebuild needs.  ``Jet_pt`` starts exactly where
    ``CorrT1METJet_rawPt`` stops, so together they are the full collection the
    production MET was built from.

    *scale* is the per-``Jet`` factor beyond the nominal JEC (smearing and/or a
    variation); ``CorrT1METJet`` jets are unsmeared, as in the prescription.
    Both collections are projected along ``phi + muonSubtrDeltaPhi``, the axis of
    the jet with its muon removed.

    Closure, on 2000 signal events: feeding the *production* jet pT through this
    same machinery reproduces the production Type-1 term to 0.001 GeV (14.165 vs
    14.166 GeV mean magnitude), which is what validates the recipe.  Rebuilding
    with the current calibration instead lands ~6 GeV from the stored PuppiMET,
    because the JEC has moved since the samples were produced: the stored jets
    carry a mean correction factor of 1.58 against 1.25 for Summer24Prompt24_V5.
    That difference is the physics, not a bug -- it is the reason to rebuild
    rather than to keep the stored value.

    Two notes on fidelity: the muon subtraction is applied here (mean
    ``muonSubtrFactor`` ~0.012, so a ~1 % effect) while the production term
    closes without it, and for PUPPI the L1 factor is exactly 1.0, so
    ``pT_L2L3Res - pT_L1`` reduces to the correction above raw.
    """
    raw_name = RAW_MET.get(met_name)
    if raw_name is None or raw_name not in events.fields:
        raise ValueError(
            f"cannot rebuild {met_name}: no raw counterpart "
            f"({raw_name or 'unknown flavour'}) in the events. Pass met=None to "
            "leave MET untouched.")

    def flat(arr):
        return np.asarray(ak.flatten(arr), dtype=np.float64)

    rho_evt = np.asarray(events.Rho.fixedGridRhoFastjetAll, dtype=np.float64)
    run_evt = (np.asarray(events.run, dtype=np.float64)
               if "run" in events.fields else None)

    jets = events.Jet
    dx, dy = _t1_delta(
        flat(jets.area), flat(jets.eta), flat(jets.phi),
        flat(jets.pt) * (1.0 - flat(jets.rawFactor)),
        flat(jets.muonSubtrFactor),
        flat(jets.chEmEF) + flat(jets.neEmEF),
        cset, jec_tag, algo, np.repeat(rho_evt, counts),
        run=None if run_evt is None else np.repeat(run_evt, counts),
        scale=scale, dphi=_musub_dphi(jets, flat))
    sum_x = ak.sum(ak.unflatten(dx, counts), axis=1)
    sum_y = ak.sum(ak.unflatten(dy, counts), axis=1)

    if "CorrT1METJet" in events.fields:
        low = events.CorrT1METJet
        n_low = np.asarray(ak.num(low))
        lx, ly = _t1_delta(
            flat(low.area), flat(low.eta), flat(low.phi),
            flat(low.rawPt), flat(low.muonSubtrFactor), flat(low.EmEF),
            cset, jec_tag, algo, np.repeat(rho_evt, n_low),
            run=None if run_evt is None else np.repeat(run_evt, n_low),
            dphi=_musub_dphi(low, flat))
        sum_x = sum_x + ak.sum(ak.unflatten(lx, n_low), axis=1)
        sum_y = sum_y + ak.sum(ak.unflatten(ly, n_low), axis=1)

    raw = events[raw_name]
    mex = raw.pt * np.cos(raw.phi) - sum_x
    mey = raw.pt * np.sin(raw.phi) - sum_y

    met = events[met_name]
    met = ak.with_field(met, np.hypot(mex, mey), "pt")
    met = ak.with_field(met, np.arctan2(mey, mex), "phi")
    return ak.with_field(events, met, met_name)


def _per_jet(values, counts):
    """Broadcast a per-event array to one entry per jet."""
    return ak.unflatten(np.repeat(np.asarray(values, dtype=np.float64), counts),
                        counts)


def _stack(cset, jec_tag, algo, *, with_junc):
    """Wrap the jsonpog JEC in coffea's correctionlib adapters.

    Only the JEC (and its total uncertainty) goes through the factory; the
    smearing uses the official ``JERSmear`` payload in :func:`_smear_jets`.
    """
    from coffea.jetmet_tools.correctionlib_adapters import (
        CorrectionLibJEC, CorrectionLibJECStack, CorrectionLibJUNC)

    key = f"{jec_tag}_L1L2L3Res_{algo}"
    if key not in cset.compound:
        available = sorted({k.rsplit("_L1L2L3Res", 1)[0]
                            for k in cset.compound if k.endswith(f"_L1L2L3Res_{algo}")})
        raise KeyError(
            f"JEC tag '{jec_tag}' not in this payload for {algo}. "
            f"Available: {available or '(none)'}. Either point "
            "$CORRECTIONLIB_DATA at a jsonpog copy that has it, or set the tag "
            "in suep_plot.jme.DEFAULTS to one of the above.")

    kwargs = {"jec": CorrectionLibJEC(cset.compound[key])}
    if with_junc:
        kwargs["junc"] = CorrectionLibJUNC([("Total", cset[f"{jec_tag}_Total_{algo}"])])
    return CorrectionLibJECStack(**kwargs)


def _smear_jets(jets, events, cset, jer_tag, algo, variation, counts):
    """Apply the official ``JERSmear`` factor to JEC-corrected *jets*.

    Follows the JME recipe: evaluate the pT resolution and its scale factor at
    the corrected pT, take the gen pT of jets matched within ``dR < 0.2`` and
    ``3 sigma`` (``-1`` otherwise, which switches ``JERSmear`` from the
    gen-matched scaling to its deterministic stochastic mode, seeded by the
    event number), then scale pT and mass by the factor.  A non-finite or
    non-positive factor falls back to ``1.0`` -- no smearing rather than a
    destroyed jet.
    """
    def flat(arr):
        return np.asarray(ak.flatten(arr), dtype=np.float64)

    pt, eta = flat(jets.pt), flat(jets.eta)
    rho = flat(jets.event_rho)
    syst = {"jer_up": "up", "jer_down": "down"}.get(variation, "nom")

    res = _evaluate(cset[f"{jer_tag}_PtResolution_{algo}"],
                    {"JetEta": eta, "JetPt": pt, "Rho": rho})
    jersf = _evaluate(cset[f"{jer_tag}_ScaleFactor_{algo}"],
                      {"JetEta": eta, "JetPt": pt, "systematic": syst})
    gen_pt = _matched_gen_pt(jets, events.GenJet, pt, res, counts)

    factor = _evaluate(_smear_cset()["JERSmear"],
                       {"JetPt": pt, "JetEta": eta, "GenPt": gen_pt, "Rho": rho,
                        "EventID": np.repeat(np.asarray(events.event, dtype=np.int64),
                                             counts),
                        "JER": res, "JERSF": jersf})
    factor = np.where(np.isfinite(factor) & (factor > 0.0), factor, 1.0)

    jets = ak.with_field(jets, ak.unflatten(pt * factor, counts), "pt")
    return ak.with_field(jets, ak.unflatten(flat(jets.mass) * factor, counts),
                         "mass")


_NAME_MAP = {
    "JetPt": "pt", "JetMass": "mass", "JetEta": "eta", "JetPhi": "phi",
    "JetA": "area", "Rho": "event_rho", "run": "run",
    "ptGenJet": "pt_gen", "ptRaw": "pt_raw", "massRaw": "mass_raw",
}


def correct_jets(
    events,
    era: str = "2024_Summer24",
    algo: str = "AK4PFPuppi",
    jec_tag: str | None = None,
    jer_tag: str | None = None,
    variation: str = "nominal",
    smear: bool = True,
    met: str | None = "PuppiMET",
):
    """Return *events* with JEC-corrected (and, for MC, JER-smeared) jets.

    The JEC comes from coffea's :class:`CorrectedJetsFactory` (fed the jsonpog
    payloads through coffea's ``correctionlib_adapters``); the JER smearing from
    the official ``JERSmear`` payload, see :func:`_smear_jets`.  What stays here
    is the wiring: resolving the payload tags, building the per-jet
    ``event_rho`` / ``run`` / raw-pT columns the factory reads through
    ``name_map``, the Type-1 MET propagation and the pT re-sort.

    Parameters
    ----------
    era       : jsonpog directory, e.g. ``2024_Summer24``.
    algo      : jet algorithm key, e.g. ``AK4PFPuppi``.
    jec_tag / jer_tag :
        payload tags (default: the era entry in :data:`DEFAULTS`).  For data,
        pass the era's ``jec_tag_data`` (``L1L2L3Res`` including the residual
        corrections) and ``smear=False``.
    variation : ``nominal``, ``jec_up``/``jec_down`` (total JEC uncertainty)
        or ``jer_up``/``jer_down`` (resolution SF variation).
    smear     : apply JER smearing (automatically off when the events carry
        no ``GenJet`` collection, i.e. data).
    met       : MET collection to Type-1-propagate the jet change into
        (skipped when absent from the events); ``None`` disables.
    """
    from coffea.jetmet_tools import CorrectedJetsFactory

    if variation not in _VARIATIONS:
        raise ValueError(f"variation must be one of {_VARIATIONS}")
    defaults = DEFAULTS.get(era, {})
    jec_tag = jec_tag or defaults.get("jec_tag")
    jer_tag = jer_tag or defaults.get("jer_tag")
    if jec_tag is None:
        raise ValueError(f"no default jec_tag for era '{era}'; pass jec_tag=...")

    cset = _load_cset(payload_path(era, "jet_jerc.json.gz"))
    jets = events.Jet
    counts = np.asarray(ak.num(jets))

    def flat(arr):
        return np.asarray(ak.flatten(arr), dtype=np.float64)

    # Columns the factory reads through _NAME_MAP.  "run" only exists in the
    # DATA compound's signature: one payload covers the whole year and picks the
    # residual IOV per event from the run number.
    # Deliberately NOT muon-subtracted here.  The jet four-momentum is the whole
    # jet, muon content included, and that is what the JEC was derived for -- so
    # jet pT, HT and the jet selections all use the full jet.  Only the MET sum
    # uses ``1 - muonSubtrFactor`` (see _t1_delta), because MET already counts
    # the muon directly and correcting it as hadronic energy would double count.
    raw = 1.0 - jets.rawFactor
    jets = ak.with_field(jets, jets.pt * raw, "pt_raw")
    jets = ak.with_field(jets, jets.mass * raw, "mass_raw")
    # NB: not "rho" -- that is a coffea vector alias for pt on JetArray.
    jets = ak.with_field(jets, _per_jet(events.Rho.fixedGridRhoFastjetAll, counts),
                         "event_rho")
    if "run" in events.fields:
        jets = ak.with_field(jets, _per_jet(events.run, counts), "run")

    do_smear = smear and jer_tag is not None and "GenJet" in events.fields
    with_junc = variation in ("jec_up", "jec_down")

    # ── JEC (coffea's factory) ────────────────────────────────────
    nominal = CorrectedJetsFactory(
        _NAME_MAP, _stack(cset, jec_tag, algo, with_junc=with_junc)).build(jets)
    corrected = nominal
    if with_junc:
        corrected = corrected.JES_Total["up" if variation == "jec_up" else "down"]

    # ── JER smearing (official JERSmear payload, MC only) ─────────
    if do_smear:
        corrected = _smear_jets(corrected, events, cset, jer_tag, algo,
                                variation, counts)

    # ── Type-1 MET, rebuilt from the raw MET ──────────────────────
    if met is not None and met in events.fields:
        # Everything the jets picked up beyond the nominal JEC -- smearing, a JES
        # variation -- has to reach the MET sum too, and it is the same
        # multiplicative factor there.
        nom_pt = flat(nominal.pt)
        scale = np.divide(flat(corrected.pt), nom_pt,
                          out=np.ones_like(nom_pt), where=nom_pt > 0)
        events = _rebuild_met(events, met, cset, jec_tag, algo, scale, counts)

    # ── write back and re-sort by the new pT ──────────────────────
    keep = [f for f in corrected.fields
            if f not in ("pt_raw", "mass_raw", "event_rho", "pt_gen")]
    new_jets = corrected[keep]
    order = ak.argsort(new_jets.pt, axis=1, ascending=False)
    return ak.with_field(events, new_jets[order], "Jet")
