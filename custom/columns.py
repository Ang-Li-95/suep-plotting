"""User-defined derived columns (coffea NanoEvents).

The processor calls ``derive(events)`` once per chunk, before filling
histograms, and uses the returned array in place of ``events``.  Add new fields
with :func:`awkward.with_field` and reference them in ``histograms.yaml`` /
``selections.yaml`` expressions as ``events.<name>``.

Return the (possibly augmented) ``events`` array.  If you don't need any derived
columns, the default pass-through below is fine.
"""

from __future__ import annotations

import awkward as ak
import numpy as np  # noqa: F401  (available for your derivations)


def derive(events):
    """Compute derived quantities and attach them as new fields on *events*.

    Parameters
    ----------
    events : NanoEvents array for the current chunk.

    Returns
    -------
    The events array, optionally with extra fields added.

    Examples
    --------
    Jet energy corrections + resolution smearing (JEC/JER).  These rescale
    the jet four-momentum, so they belong here (before selections and fills),
    not in corrections.yaml.  Needs cvmfs jsonpog-integration or
    $CORRECTIONLIB_DATA; see suep_plot/jme.py for the full prescription::

        from suep_plot.jme import correct_jets
        events = correct_jets(events)              # 2024 Summer24 MC defaults
        # systematics:  correct_jets(events, variation="jec_up")
        #               (jec_down / jer_up / jer_down)
        # data:         correct_jets(events, jec_tag="<era>_DATA", smear=False)
        # afterwards every events.Jet expression uses corrected jets
        # (MET is not propagated)

    Scalar HT as an event-level field::

        events = ak.with_field(events, ak.sum(events.Jet.pt, axis=1), "HT")
        # then in histograms.yaml:  expression: "events.HT"

    Leading-dimuon invariant mass (Muon carries Lorentz-vector behaviour)::

        pair = events.Muon[ak.num(events.Muon) >= 2]
        mll = (pair[:, 0] + pair[:, 1]).mass
        # broadcast back to full length as needed, or build an event-level field.

    DBSCAN clustering on CSC rechits (requires scikit-learn and the rechit
    branches present in the file)::

        from sklearn.cluster import DBSCAN
        eta, phi = events.cscRechits.eta, events.cscRechits.phi
        nclu = np.zeros(len(events), dtype=np.int32)
        for i in range(len(events)):
            coords = np.column_stack([ak.to_numpy(eta[i]), ak.to_numpy(phi[i])])
            if len(coords) >= 2:
                labels = DBSCAN(eps=0.2, min_samples=50).fit_predict(coords)
                nclu[i] = len(set(labels) - {-1})
        events = ak.with_field(events, nclu, "csc_dbscan_nclusters")
    """
    return events
