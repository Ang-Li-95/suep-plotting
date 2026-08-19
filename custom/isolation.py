"""Cluster isolation: dR from a cluster to the nearest prompt object.

Which reconstructed objects count as prompt activity is config, not code (the
``iso_objects`` parameter): :func:`_selected_objects` evaluates one
``columns.yaml`` selection expression and :func:`_dr_to_nearest` measures the
dR from every cluster centroid to the closest survivor.  :func:`_jet_id`
supplies the official jsonpog jet-ID decision those expressions can call, since
2024 NanoAOD no longer stores ``Jet_jetId``.
"""

from __future__ import annotations

import functools

import awkward as ak
import numpy as np

from .params import PARAMS


# drMuon / drJet in events with no reconstructed muon / jet at all.  Such a
# cluster is maximally isolated, so the sentinel is larger than any physical dR
# (which cannot exceed ~2*pi) and passes every isolation cut without the cut
# expression needing an isnan leg.  It lands in the overflow of the dR
# histograms, which fold it into their last bin ('flow: sum', the default), so
# the count of clusters with no prompt object stays visible on the plot.
NO_OBJECT_DR = 999.0


def _dr_to_nearest(clusters, objects):
    """dR from each cluster centroid to the closest object in the event.

    :data:`NO_OBJECT_DR` in events with no such object (``ak.min`` over an empty
    list): a cluster with no muon in the event is maximally isolated, so the
    sentinel is deliberately larger than any real dR and passes every isolation
    cut.  It overflows the dR axis, and those histograms keep the default
    ``flow: sum``, so it shows up in their last bin.  Written out by hand
    because the cluster collection is a plain record array with no vector
    behaviour, so ``.nearest()`` is unavailable.
    """
    deta = clusters.eta[:, :, None] - objects.eta[:, None, :]
    dphi = (clusters.phi[:, :, None] - objects.phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    return ak.fill_none(ak.min(np.sqrt(deta ** 2 + dphi ** 2), axis=2), NO_OBJECT_DR)


_SAFE_BUILTINS = {"abs": abs, "len": len, "min": min, "max": max}


def _selected_objects(events, selection):
    """Objects passing *selection*, or None when the collection is absent.

    *selection* is ``{"collection": <NanoAOD collection>, "expression": <per-object
    boolean>}``, exactly as written in ``columns.yaml``.  The expression sees
    ``obj`` (the collection), ``events``/``ev``, ``ak``, ``np``, the safe builtins
    and ``jet_id``; nothing about any particular object type is hard-coded here.
    """
    collection = selection["collection"]
    if collection not in events.fields:
        return None

    objects = events[collection]
    try:
        keep = eval(  # noqa: S307 - trusted config expressions
            selection["expression"],
            {"__builtins__": _SAFE_BUILTINS},
            {"obj": objects, "events": events, "ev": events, "ak": ak, "np": np,
             "jet_id": lambda obj, level: _jet_id(obj, str(level).lower(),
                                                  PARAMS["jerc_era"])},
        )
    except Exception as exc:
        raise ValueError(
            f"columns.yaml: iso_objects expression for '{collection}' failed: "
            f"{selection['expression']!r}\n  {type(exc).__name__}: {exc}") from exc

    return objects[keep]


@functools.lru_cache(maxsize=None)
def _jet_id_evaluator(era, algo, level):
    """The official jsonpog jet-ID evaluator (cached; opening the gz is slow)."""
    import correctionlib

    from suep_plot.jme import payload_path

    key = {"tight": f"{algo}_Tight",
           "tightlepveto": f"{algo}_TightLeptonVeto"}[level]
    # Same resolution as the JEC, so the ID and the calibration come from one
    # campaign directory rather than drifting apart.
    return correctionlib.CorrectionSet.from_file(
        payload_path(era, "jetid.json.gz"))[key]


def _jet_id(jets, level, era, algo="AK4PUPPI"):
    """Per-jet ID decision as a jagged boolean.

    2024 NanoAOD dropped ``Jet_jetId``; the JME prescription is to recompute it
    from the PF energy fractions and multiplicities, which the official
    ``jetid.json.gz`` does -- so the thresholds live in the central payload
    rather than being copied into this file.
    """
    counts = ak.num(jets)
    flat = {name: ak.to_numpy(ak.flatten(jets[field]))
            for name, field in (("eta", "eta"), ("chHEF", "chHEF"),
                                ("neHEF", "neHEF"), ("chEmEF", "chEmEF"),
                                ("neEmEF", "neEmEF"), ("muEF", "muEF"),
                                ("chMultiplicity", "chMultiplicity"),
                                ("neMultiplicity", "neMultiplicity"),
                                ("multiplicity", "nConstituents"))}
    for key in ("chMultiplicity", "neMultiplicity", "multiplicity"):
        flat[key] = flat[key].astype(np.int32)

    evaluator = _jet_id_evaluator(era, algo, level)
    passed = evaluator.evaluate(*[flat[i.name] for i in evaluator.inputs])
    return ak.unflatten(passed.astype(bool), counts)
