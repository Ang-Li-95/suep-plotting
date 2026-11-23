"""suep_plot.overlay: overlay.yaml and the MC cleaning of the muon rechits."""

import math

import awkward as ak
import numpy as np
import pytest

from suep_plot.overlay import (CleaningCuts, OverlayConfig, ZeroBiasSource,
                               clean_mc, load_overlay_config)

STUDY_CUTS = CleaningCuts(csc_tpeak=(-30, 40), rpc_bx=(0, 1), dr=0.4, genjet_pt=10)

VALID_YAML = """
mc_clean: {csc_tpeak: [-30, 40], rpc_bx: [0, 1], dr: 0.4, genjet_pt: 10}
add_zerobias: false
"""

WITH_ZEROBIAS = """
mc_clean: {csc_tpeak: [-30, 40], rpc_bx: [0, 1], dr: 0.4, genjet_pt: 10}
add_zerobias: true
zerobias:
  pool: zerobias_2024
  lumi_mask: lumimasks/golden.json
  data_clean: {dr: 0.4}
"""


# ── overlay.yaml ──────────────────────────────────────────────────


def _load(tmp_path, text):
    path = tmp_path / "overlay.yaml"
    path.write_text(text)
    return load_overlay_config(path)


def test_a_valid_file_loads(tmp_path):
    assert _load(tmp_path, VALID_YAML) == OverlayConfig(STUDY_CUTS, zerobias=None)


def test_the_zerobias_section_loads(tmp_path):
    source = ZeroBiasSource("zerobias_2024", "lumimasks/golden.json", data_clean_dr=0.4)
    assert _load(tmp_path, WITH_ZEROBIAS) == OverlayConfig(STUDY_CUTS, source)


def test_data_clean_can_be_switched_off(tmp_path):
    config = _load(tmp_path, WITH_ZEROBIAS.replace("{dr: 0.4}", "false"))
    assert config.zerobias.data_clean_dr is None


def test_no_file_means_no_overlay(tmp_path):
    assert load_overlay_config(tmp_path / "overlay.yaml") is None


@pytest.mark.parametrize("text, message", [
    (VALID_YAML + "seed: 3\n", "unknown key"),
    ("add_zerobias: false\n", "missing key"),
    (VALID_YAML.replace("dr: 0.4, ", ""), "missing key"),
    (VALID_YAML.replace("[-30, 40]", "[40, -30]"), "csc_tpeak"),
    (VALID_YAML.replace("[0, 1]", "[0.5]"), "rpc_bx"),
    (VALID_YAML.replace("dr: 0.4", "dr: 0"), "dr must be"),
    (VALID_YAML.replace("add_zerobias: false", "add_zerobias: 'no'"), "true or false"),
    (VALID_YAML.replace("add_zerobias: false", "add_zerobias: true"), "no 'zerobias:'"),
    (WITH_ZEROBIAS.replace("add_zerobias: true", "add_zerobias: false"), "is given but"),
    (WITH_ZEROBIAS.replace("  pool: zerobias_2024\n", ""), "missing key"),
    (WITH_ZEROBIAS.replace("{dr: 0.4}", "true"), "data_clean must be"),
    (WITH_ZEROBIAS.replace("{dr: 0.4}", "{dr: -1}"), "positive"),
])
def test_a_bad_file_is_fatal(tmp_path, text, message):
    with pytest.raises(SystemExit, match=message):
        _load(tmp_path, text)


# ── the cleaning ──────────────────────────────────────────────────

# An endcap position (CSC, RPC endcap) and a barrel one (DT), in cm.
ENDCAP_XYZ = (200.0, 0.0, 700.0)
BARREL_XYZ = (420.0, 0.0, 100.0)


def _position(x, y, z):
    r = math.hypot(x, y)
    return {"X": x, "Y": y, "Z": z, "Eta": math.asinh(z / r), "Phi": math.atan2(y, x)}


POSITION = ("X", "Y", "Z", "Eta", "Phi")
INTEGER_FIELDS = ("Bx", "pdgId", "status")


def _collection(rows, fields):
    """One event's collection: a list of field dicts (possibly empty)."""
    def column(field):
        dtype = np.int32 if field in INTEGER_FIELDS else np.float32
        values = np.asarray([row[field] for row in rows], dtype=dtype)
        return ak.unflatten(values, [len(rows)])
    return ak.zip({f: column(f) for f in fields})


def _events(csc=(), dt=(), rpc=(), genjets=(), gen=()):
    """One event: rechits given as (position, extra fields), gen objects as dicts."""
    def hits(entries):
        return [{**_position(*pos), **extra} for pos, extra in entries]

    return ak.zip({
        "cscRechits": _collection(hits(csc), (*POSITION, "Tpeak")),
        "dtRecHits": _collection(hits(dt), POSITION),
        "rpcRecHits": _collection(hits(rpc), (*POSITION, "Bx")),
        "GenJet": _collection(genjets, ("pt", "eta", "phi")),
        "GenPart": _collection(gen, ("pdgId", "status", "pt", "eta", "phi")),
    }, depth_limit=1)


def _jet_at(xyz, pt=50.0):
    position = _position(*xyz)
    return {"pt": pt, "eta": position["Eta"], "phi": position["Phi"]}


def _kept(events, collection):
    return ak.to_list(events[f"{collection}MC"].cleanKeep[0])


def test_csc_time_window_is_open():
    tpeaks = [-30.0, -29.9, 39.9, 40.0]
    events = _events(csc=[(ENDCAP_XYZ, {"Tpeak": t}) for t in tpeaks],
                     genjets=[_jet_at(ENDCAP_XYZ)])

    events = clean_mc(events, STUDY_CUTS)

    assert _kept(events, "cscRechits") == [False, True, True, False]


def test_rpc_keeps_only_the_listed_bunch_crossings():
    events = _events(rpc=[(BARREL_XYZ, {"Bx": bx}) for bx in (-1, 0, 1, 2)],
                     genjets=[_jet_at(BARREL_XYZ)])

    events = clean_mc(events, STUDY_CUTS)

    assert _kept(events, "rpcRecHits") == [False, True, True, False]


def test_dt_has_no_timing_cut_only_the_dr_one():
    far = (0.0, 420.0, 100.0)          # 90 degrees away from the jet
    events = _events(dt=[(BARREL_XYZ, {}), (far, {})], genjets=[_jet_at(BARREL_XYZ)])

    events = clean_mc(events, STUDY_CUTS)

    assert _kept(events, "dtRecHits") == [True, False]


@pytest.mark.parametrize("jet_pt, counts", [(10.0, False), (10.1, True)])
def test_gen_jets_count_strictly_above_the_threshold(jet_pt, counts):
    events = _events(dt=[(BARREL_XYZ, {})], genjets=[_jet_at(BARREL_XYZ, pt=jet_pt)])

    events = clean_mc(events, STUDY_CUTS)

    assert _kept(events, "dtRecHits") == [counts]


def test_a_reachable_gen_muon_keeps_its_hits():
    position = _position(*BARREL_XYZ)
    muon = {"pdgId": 13, "status": 1, "pt": 50.0, "eta": position["Eta"],
            "phi": position["Phi"]}
    events = _events(dt=[(BARREL_XYZ, {})], gen=[muon])

    events = clean_mc(events, STUDY_CUTS)

    assert _kept(events, "dtRecHits") == [True]


def test_the_cleaned_collection_holds_the_kept_hits_flagged_as_simulation():
    events = _events(csc=[(ENDCAP_XYZ, {"Tpeak": 0.0}), (ENDCAP_XYZ, {"Tpeak": 90.0})],
                     genjets=[_jet_at(ENDCAP_XYZ)])

    events = clean_mc(events, STUDY_CUTS)

    assert ak.to_list(events.cscRechits.Tpeak[0]) == [0.0]
    assert ak.to_list(events.cscRechits.fromZeroBias[0]) == [False]
    assert ak.num(events.cscRechitsMC)[0] == 2      # the input is kept whole


def test_the_cleaning_needs_the_gen_jets():
    events = _events(dt=[(BARREL_XYZ, {})])
    events = ak.zip({f: events[f] for f in events.fields if f != "GenJet"},
                    depth_limit=1)

    with pytest.raises(ValueError, match="GenJet"):
        clean_mc(events, STUDY_CUTS)
