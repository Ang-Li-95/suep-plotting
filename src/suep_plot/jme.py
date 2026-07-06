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

What it does, following the JME prescription:

1. undo the NanoAOD correction: ``pt_raw = pt * (1 - rawFactor)``;
2. apply the compound ``L1L2L3Res`` JEC from the jsonpog ``jet_jerc.json.gz``;
3. (MC) smear with the official ``JERSmear`` helper — gen-matched scaling for
   jets with a valid ``genJetIdx`` within ``dR < 0.2`` and ``|pt - pt_gen| <
   3 sigma``, deterministic stochastic smearing otherwise (seeded from the
   event number, so results are reproducible);
4. re-sort jets by the new pT.

``pt`` and ``mass`` are replaced in place; MET is *not* propagated.
Correction files resolve like ``corrections.yaml`` (``$CORRECTIONLIB_DATA``,
then cvmfs jsonpog-integration).
"""

from __future__ import annotations

import functools

import awkward as ak
import numpy as np

from .corrections import _resolve_file

# JME ships Summer23BPix JER payloads inside the 2024_Summer24 file until the
# Summer24 JER is released; these defaults match the file contents.
DEFAULTS = {
    "2024_Summer24": {
        "jec_tag": "Summer24Prompt24_V1_MC",
        "jer_tag": "Summer23BPixPrompt23_RunD_JRV1_MC",
    },
}

_VARIATIONS = ("nominal", "jec_up", "jec_down", "jer_up", "jer_down")


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


def correct_jets(
    events,
    era: str = "2024_Summer24",
    algo: str = "AK4PFPuppi",
    jec_tag: str | None = None,
    jer_tag: str | None = None,
    variation: str = "nominal",
    smear: bool = True,
):
    """Return *events* with JEC-corrected (and, for MC, JER-smeared) jets.

    Parameters
    ----------
    era       : jsonpog directory, e.g. ``2024_Summer24``.
    algo      : jet algorithm key, e.g. ``AK4PFPuppi``.
    jec_tag / jer_tag :
        payload tags (default: the era entry in :data:`DEFAULTS`).  For data,
        pass the run-specific ``*_DATA`` jec_tag and ``smear=False``.
    variation : ``nominal``, ``jec_up``/``jec_down`` (total JEC uncertainty)
        or ``jer_up``/``jer_down`` (resolution SF variation).
    smear     : apply JER smearing (automatically off when the events carry
        no ``GenJet`` collection, i.e. data).
    """
    if variation not in _VARIATIONS:
        raise ValueError(f"variation must be one of {_VARIATIONS}")
    defaults = DEFAULTS.get(era, {})
    jec_tag = jec_tag or defaults.get("jec_tag")
    jer_tag = jer_tag or defaults.get("jer_tag")
    if jec_tag is None:
        raise ValueError(f"no default jec_tag for era '{era}'; pass jec_tag=...")

    cset = _load_cset(f"auto:JME/{era}/jet_jerc.json.gz")
    jets = events.Jet
    counts = np.asarray(ak.num(jets))

    def flat(arr):
        return np.asarray(ak.flatten(arr), dtype=np.float64)

    pt, eta, phi, mass = flat(jets.pt), flat(jets.eta), flat(jets.phi), flat(jets.mass)
    raw = 1.0 - flat(jets.rawFactor)
    pt_raw, mass_raw = pt * raw, mass * raw
    rho = np.repeat(np.asarray(events.Rho.fixedGridRhoFastjetAll, dtype=np.float64),
                    counts)

    # ── JEC ───────────────────────────────────────────────────────
    jec = cset.compound[f"{jec_tag}_L1L2L3Res_{algo}"]
    sf = _evaluate(jec, {"JetA": flat(jets.area), "JetEta": eta, "JetPhi": phi,
                         "JetPt": pt_raw, "Rho": rho})
    pt_corr, mass_corr = pt_raw * sf, mass_raw * sf

    if variation in ("jec_up", "jec_down"):
        unc = _evaluate(cset[f"{jec_tag}_Total_{algo}"],
                        {"JetEta": eta, "JetPt": pt_corr})
        shift = 1.0 + unc if variation == "jec_up" else 1.0 - unc
        pt_corr, mass_corr = pt_corr * shift, mass_corr * shift

    # ── JER smearing (MC only) ────────────────────────────────────
    if smear and jer_tag is not None and "GenJet" in events.fields:
        syst = {"jer_up": "up", "jer_down": "down"}.get(variation, "nom")
        res = _evaluate(cset[f"{jer_tag}_PtResolution_{algo}"],
                        {"JetEta": eta, "JetPt": pt_corr, "Rho": rho})
        jersf = _evaluate(cset[f"{jer_tag}_ScaleFactor_{algo}"],
                          {"JetEta": eta, "JetPt": pt_corr, "systematic": syst})
        gen_pt = _matched_gen_pt(jets, events.GenJet, pt_corr, res, counts)
        # JERSmear declares EventID as int (used to seed the deterministic RNG).
        event_id = np.repeat(np.asarray(events.event, dtype=np.int64), counts)

        smear_cset = _load_cset("auto:JME/jer_smear.json.gz")
        factor = _evaluate(smear_cset["JERSmear"],
                           {"JetPt": pt_corr, "JetEta": eta, "GenPt": gen_pt,
                            "Rho": rho, "EventID": event_id,
                            "JER": res, "JERSF": jersf})
        # Guard against negative smears for extreme stochastic draws.
        factor = np.maximum(factor, 0.0)
        pt_corr, mass_corr = pt_corr * factor, mass_corr * factor

    # ── write back and re-sort by the new pT ──────────────────────
    new_jets = ak.with_field(jets, ak.unflatten(pt_corr, counts), "pt")
    new_jets = ak.with_field(new_jets, ak.unflatten(mass_corr, counts), "mass")
    order = ak.argsort(new_jets.pt, axis=1, ascending=False)
    return ak.with_field(events, new_jets[order], "Jet")
