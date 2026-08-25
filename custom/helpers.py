"""Assorted helpers for the derived columns: object selection, isolation, jet ID.

Which reconstructed objects count as prompt activity is config, not code, and it
is config in the ordinary place: :func:`_selected_objects` evaluates one
**selections.yaml** entry, so a prompt-object cut is written exactly like every
other object cut in the package.  :func:`_cluster_isolation` measures the dR
from every cluster centroid to the closest survivor of
:data:`ISO_SELECTIONS`.

:func:`_jet_id` is the official jsonpog jet-ID decision, which 2024 NanoAOD no
longer stores as ``Jet_jetId``.  The ``jet_id`` step attaches it to
``events.Jet`` as a plain column, so a selection asks for
``events.Jet.tightLepVetoId`` rather than calling a function that only exists
inside one expression scope.
"""

from __future__ import annotations

import functools

import awkward as ak
import numpy as np



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


# The selections.yaml entries the cluster isolation is measured against, and
# the dR field each one produces.  Both are object-level selections over a
# prompt collection; see docs/derived-columns.md.
ISO_SELECTIONS = {"drMuon": "muon_sel_foriso", "drJet": "jet_sel_foriso"}


def _selected_objects(events, name, selection):
    """Objects passing a selections.yaml entry, or None if its collection is absent.

    *selection* is the entry as written -- a ``collection`` naming what the
    per-object mask is over, and an ``expression`` in the ordinary config
    dialect (``events.Muon.pt > 10``, the same one histograms.yaml uses).  A
    missing collection is not an error: a file with no ``Muon`` branch simply
    has no muon to be near, which :func:`_cluster_isolation` turns into the
    "maximally isolated" sentinel.
    """
    collection = selection.get("collection")
    if not collection:
        raise ValueError(
            f"selections.yaml: '{name}' is used to select objects, so it needs a "
            "'collection:' naming the collection its expression masks (e.g. Muon)")
    if collection not in events.fields:
        return None

    try:
        keep = eval(  # noqa: S307 - trusted config expressions
            selection["expression"],
            {"__builtins__": _SAFE_BUILTINS},
            {"events": events, "ev": events, "ak": ak, "np": np},
        )
    except Exception as exc:
        raise ValueError(
            f"selections.yaml: '{name}' failed on collection '{collection}': "
            f"{selection['expression']!r}\n  {type(exc).__name__}: {exc}") from exc

    return events[collection][keep]


def _cluster_isolation(clusters, objects):
    """Attach one dR field per :data:`ISO_SELECTIONS` entry to *clusters*.

    *objects* maps the dR field name to the selected collection (or None when
    that collection is absent from the file, in which case the field is
    skipped -- a config referencing it then fails at expression validation
    rather than filling zeros).
    """
    for field, objs in objects.items():
        if objs is not None:
            clusters = ak.with_field(clusters, _dr_to_nearest(clusters, objs), field)
    return clusters


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
