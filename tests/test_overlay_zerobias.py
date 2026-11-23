"""The ZeroBias half of the overlay: pairing plans, picking, DataClean, merging.

The ZeroBias inputs are small ROOT files written here in the NanoAOD layout
(``ncscRechits``, ``cscRechits_X``, ...), so the picker reads them exactly as
it reads the real pool.  Every ZeroBias event carries a unique ``event``
number, which is how the tests see which events were picked.
"""

import json
import math

import awkward as ak
import numpy as np
import pytest
import uproot

from suep_plot import overlay_pool
from suep_plot.histdefs import Selection
from suep_plot.overlay import (CleaningCuts, Overlay, OverlayConfig, ZeroBiasSource,
                               add_zerobias, clean_mc, clean_zerobias)
from suep_plot.overlay_pool import (PoolPlan, ZeroBiasPicker, build_sample_plan,
                                    load_pool, load_sample_plan, plan_dir,
                                    prepare_plans, shuffle_entries)

CUTS = CleaningCuts(csc_tpeak=(-30, 40), rpc_bx=(0, 1), dr=0.4, genjet_pt=10)
RECHITS = ("cscRechits", "dtRecHits", "rpcRecHits")
# Run 1, lumi sections 1-3 are certified; lumi section 4 is not.
CERTIFIED_LUMIS = {"1": [[1, 3]]}
EVENTS_PER_LUMI = 3
LUMIS_PER_FILE = 4          # 12 events per file, 9 of them certified


def _position(x, y, z):
    r = math.hypot(x, y)
    return {"X": x, "Y": y, "Z": z, "Eta": math.asinh(z / r), "Phi": math.atan2(y, x)}


ENDCAP = _position(200.0, 0.0, 700.0)
BARREL = _position(420.0, 0.0, 100.0)
FAR_BARREL = _position(0.0, 420.0, 100.0)    # 90 degrees in phi from BARREL


def _jagged(values, dtype, counts):
    return ak.unflatten(np.asarray(values, dtype=dtype), counts)


def _hits(position, n_events, extra=None):
    """One rechit per event at *position*, with *extra* fields."""
    columns = {f: _jagged([v] * n_events, np.float32, [1] * n_events)
               for f, v in position.items()}
    for field, (value, dtype) in (extra or {}).items():
        columns[field] = _jagged([value] * n_events, dtype, [1] * n_events)
    return ak.zip(columns)


def _write_zerobias_file(path, first_event_number):
    """A ZeroBias file: run 1, LUMIS_PER_FILE lumi sections, one rechit per
    system per event, and one jet and one muon pointing at the DT/RPC hit."""
    n = EVENTS_PER_LUMI * LUMIS_PER_FILE
    jet = ak.zip({"pt": _jagged([50.0] * n, np.float32, [1] * n),
                  "eta": _jagged([BARREL["Eta"]] * n, np.float32, [1] * n),
                  "phi": _jagged([BARREL["Phi"]] * n, np.float32, [1] * n)})
    muon = ak.zip({"pt": _jagged([50.0] * n, np.float32, [1] * n),
                   "eta": _jagged([ENDCAP["Eta"]] * n, np.float32, [1] * n),
                   "phi": _jagged([ENDCAP["Phi"]] * n, np.float32, [1] * n),
                   "charge": _jagged([1] * n, np.int32, [1] * n)})
    data = {
        "run": np.ones(n, dtype=np.uint32),
        "luminosityBlock": np.repeat(np.arange(1, LUMIS_PER_FILE + 1),
                                     EVENTS_PER_LUMI).astype(np.uint32),
        "event": np.arange(first_event_number, first_event_number + n, dtype=np.uint64),
        "cscRechits": _hits(ENDCAP, n, {"Tpeak": (0.0, np.float32)}),
        "dtRecHits": _hits(FAR_BARREL, n),
        "rpcRecHits": _hits(BARREL, n, {"Bx": (0, np.int32)}),
        "Jet": jet,
        "Muon": muon,
    }
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", {k: v.type if isinstance(v, ak.Array) else v.dtype
                                    for k, v in data.items()},
                         field_name=lambda outer, inner: f"{outer}_{inner}",
                         counter_name=lambda counted: "n" + counted)
        root_file["Events"].extend(data)


@pytest.fixture
def zerobias_files(tmp_path):
    """Six ZeroBias files (events numbered 1000 per file) and the lumi mask."""
    paths = []
    for i in range(6):
        path = tmp_path / f"zb_{i}.root"
        _write_zerobias_file(str(path), first_event_number=1000 * i)
        paths.append(str(path))
    mask = tmp_path / "golden.json"
    mask.write_text(json.dumps(CERTIFIED_LUMIS))
    return paths, str(mask)


def _pool(files, mask, seed=12345):
    return PoolPlan("zb_test", mask, "", seed, tuple(files))


def _picker(pool, plans, collections=RECHITS):
    return ZeroBiasPicker.in_memory(pool, plans, collections)


def _event_numbers(partners):
    return [int(e) for e in partners.event]


# ── the plan ──────────────────────────────────────────────────────


def test_each_mc_file_gets_its_own_block_of_pool_files():
    pool = _pool([f"zb_{i}" for i in range(10)], "mask.json")
    plan = build_sample_plan("qcd", ["a.root", "b.root", "c.root"], pool)

    blocks = [set(block) for block in plan.blocks]
    assert [len(b) for b in blocks] == [3, 3, 3]          # k = 10 // 3
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])


def test_the_shuffle_depends_on_the_run_seed_and_the_sample():
    pool = _pool([f"zb_{i}" for i in range(50)], "mask.json")
    files = ["a.root", "b.root"]

    same = build_sample_plan("qcd", files, pool)
    assert build_sample_plan("qcd", files, pool) == same
    assert build_sample_plan("wlnu", files, pool).blocks != same.blocks
    other_run = _pool(pool.files, "mask.json", seed=pool.seed + 1)
    assert build_sample_plan("qcd", files, other_run).blocks != same.blocks


def test_a_sample_larger_than_the_pool_is_refused():
    with pytest.raises(SystemExit, match="ZeroBias files of its own"):
        build_sample_plan("qcd", ["a", "b", "c"], _pool(["zb_0", "zb_1"], "m"))


@pytest.fixture
def fake_pool_listing(monkeypatch, zerobias_files):
    """prepare_plans with the pool listed from the test files, and a clock that ticks."""
    files, mask = zerobias_files
    monkeypatch.setattr(overlay_pool, "pool_definition", lambda name: {"files": files})
    monkeypatch.setattr("suep_plot.inputs.resolve_sample_files",
                        lambda cfg: list(cfg["files"]))
    ticks = iter(range(1_000_000, 2_000_000, 1000))
    monkeypatch.setattr(overlay_pool.time, "time_ns", lambda: next(ticks))
    return files, mask


def test_plans_are_made_once_and_then_frozen(tmp_path, fake_pool_listing):
    files, mask = fake_pool_listing
    run = tmp_path / "run"

    prepare_plans(run, "zb_test", mask, {"qcd": ["a.root", "b.root"]})
    first_pool = load_pool(plan_dir(run))
    first_plan = load_sample_plan(plan_dir(run), "qcd")
    prepare_plans(run, "zb_test", mask, {"qcd": ["a.root", "b.root", "c.root"]})

    assert load_pool(plan_dir(run)) == first_pool
    assert load_sample_plan(plan_dir(run), "qcd") == first_plan
    assert first_pool.files == tuple(files)


def test_a_new_run_directory_gets_a_new_seed(tmp_path, fake_pool_listing):
    _, mask = fake_pool_listing
    prepare_plans(tmp_path / "run_a", "zb_test", mask, {"qcd": ["a.root"]})
    prepare_plans(tmp_path / "run_b", "zb_test", mask, {"qcd": ["a.root"]})

    assert (load_pool(plan_dir(tmp_path / "run_a")).seed
            != load_pool(plan_dir(tmp_path / "run_b")).seed)


def test_a_plan_made_for_another_pool_is_refused(tmp_path, fake_pool_listing):
    _, mask = fake_pool_listing
    prepare_plans(tmp_path / "run", "zb_test", mask, {"qcd": ["a.root"]})
    with pytest.raises(SystemExit, match="new output directory"):
        prepare_plans(tmp_path / "run", "other_pool", mask, {"qcd": ["a.root"]})


def test_a_file_missing_from_the_plan_is_refused():
    plan = build_sample_plan("qcd", ["a.root"], _pool(["zb_0", "zb_1"], "m"))
    with pytest.raises(ValueError, match="not in the frozen pairing plan"):
        plan.block_of("new.root")


# ── the in-file shuffle ───────────────────────────────────────────


def test_the_in_file_shuffle_is_a_seeded_permutation():
    entries = np.arange(0, 200, 2)
    shuffled = shuffle_entries(entries, seed=7, pool_index=3)

    assert sorted(shuffled.tolist()) == entries.tolist()
    assert shuffled.tolist() != entries.tolist()
    assert np.array_equal(shuffle_entries(entries, seed=7, pool_index=3), shuffled)
    assert not np.array_equal(shuffle_entries(entries, seed=8, pool_index=3), shuffled)
    assert not np.array_equal(shuffle_entries(entries, seed=7, pool_index=4), shuffled)


# ── picking ───────────────────────────────────────────────────────


@pytest.fixture
def two_file_sample(zerobias_files):
    """Sample 'qcd' with two MC files: three ZeroBias files (27 certified
    events) per MC file."""
    files, mask = zerobias_files
    pool = _pool(files, mask)
    plan = build_sample_plan("qcd", ["mc_a.root", "mc_b.root"], pool)
    return _picker(pool, {"qcd": plan}), pool


def test_partners_are_certified_and_never_repeated_within_a_file(two_file_sample):
    picker, _ = two_file_sample
    partners, reused = picker.pick("qcd", "mc_a.root", 0, 27)

    numbers = _event_numbers(partners)
    assert len(set(numbers)) == 27 and not reused.any()
    assert all(int(lumi) <= 3 for lumi in partners.luminosityBlock)   # certified


def test_two_mc_files_of_a_sample_share_no_partner(two_file_sample):
    picker, _ = two_file_sample
    a, _ = picker.pick("qcd", "mc_a.root", 0, 27)
    b, _ = picker.pick("qcd", "mc_b.root", 0, 27)
    assert not set(_event_numbers(a)) & set(_event_numbers(b))


def test_chunking_does_not_change_any_pair(two_file_sample):
    picker, _ = two_file_sample
    whole, _ = picker.pick("qcd", "mc_a.root", 0, 20)
    first, _ = picker.pick("qcd", "mc_a.root", 0, 7)
    rest, _ = picker.pick("qcd", "mc_a.root", 7, 20)
    assert _event_numbers(first) + _event_numbers(rest) == _event_numbers(whole)


def test_events_inside_a_zerobias_file_are_shuffled(two_file_sample):
    picker, _ = two_file_sample
    first_file, _ = picker.pick("qcd", "mc_a.root", 0, 9)   # one file's certified events
    numbers = _event_numbers(first_file)
    assert numbers != sorted(numbers)
    assert len({n // 1000 for n in numbers}) == 1           # all from that one file


def test_a_file_that_outgrows_its_block_reuses_and_says_so(two_file_sample):
    picker, _ = two_file_sample
    partners, reused = picker.pick("qcd", "mc_a.root", 25, 30)

    assert reused.tolist() == [False, False, True, True, True]
    first, _ = picker.pick("qcd", "mc_a.root", 0, 3)
    assert _event_numbers(partners)[2:] == _event_numbers(first)   # wrapped around


def test_the_picker_reloads_its_plans_after_pickling(tmp_path, fake_pool_listing):
    import pickle

    _, mask = fake_pool_listing
    prepare_plans(tmp_path / "run", "zb_test", mask, {"qcd": ["mc_a.root"]})
    picker = ZeroBiasPicker(plan_dir(tmp_path / "run"), RECHITS)
    expected, _ = picker.pick("qcd", "mc_a.root", 0, 5)

    clone = pickle.loads(pickle.dumps(picker))
    assert clone._pool is None and clone._certified == {}
    assert _event_numbers(clone.pick("qcd", "mc_a.root", 0, 5)[0]) == _event_numbers(expected)


# ── DataClean and the merge ───────────────────────────────────────


SELECTIONS = {
    "jet_sel_foriso": Selection(expression="events.Jet.pt > 20", collection="Jet"),
    "muon_sel_foriso": Selection(expression="events.Muon.pt > 10", collection="Muon"),
}


class _Settings(tuple):
    """Stands in for custom.params.Settings: only _replace is called on it."""

    def _replace(self, **kwargs):
        return self


def _no_derive(events, settings, selections):
    return events


def test_data_clean_drops_hits_near_the_zerobias_jets_and_muons(two_file_sample):
    picker, pool = two_file_sample
    picker = _picker(pool, picker._plans, collections=(*RECHITS, "Jet", "Muon"))
    partners, _ = picker.pick("qcd", "mc_a.root", 0, 4)

    cleaned = clean_zerobias(partners, 0.4, SELECTIONS, _no_derive, _Settings())

    keep = {c: ak.to_list(ak.flatten(cleaned[c].dataCleanKeep)) for c in RECHITS}
    assert keep["rpcRecHits"] == [False] * 4      # on the jet axis
    assert keep["dtRecHits"] == [True] * 4        # far from both
    # the muon (pT 50, nearly straight) points at the CSC hit
    assert keep["cscRechits"] == [False] * 4


def test_data_clean_ignores_objects_its_selections_reject(two_file_sample):
    picker, pool = two_file_sample
    picker = _picker(pool, picker._plans, collections=(*RECHITS, "Jet", "Muon"))
    partners, _ = picker.pick("qcd", "mc_a.root", 0, 4)
    strict = {**SELECTIONS,
              "jet_sel_foriso": Selection(expression="events.Jet.pt > 100",
                                          collection="Jet")}

    cleaned = clean_zerobias(partners, 0.4, strict, _no_derive, _Settings())

    assert ak.to_list(ak.flatten(cleaned.rpcRecHits.dataCleanKeep)) == [True] * 4


def _mc_events(n):
    """n simulated events: one in-time CSC hit on a gen jet, one far DT hit."""
    genjet = ak.zip({"pt": _jagged([50.0] * n, np.float32, [1] * n),
                     "eta": _jagged([ENDCAP["Eta"]] * n, np.float32, [1] * n),
                     "phi": _jagged([ENDCAP["Phi"]] * n, np.float32, [1] * n)})
    gen = ak.zip({f: _jagged([], dtype, [0] * n) for f, dtype in
                  (("pdgId", np.int32), ("status", np.int32), ("pt", np.float32),
                   ("eta", np.float32), ("phi", np.float32))})
    return ak.zip({
        "cscRechits": _hits(ENDCAP, n, {"Tpeak": (0.0, np.float32)}),
        "dtRecHits": _hits(FAR_BARREL, n),
        "rpcRecHits": _hits(BARREL, n, {"Bx": (5, np.int32)}),
        "GenJet": genjet, "GenPart": gen,
    }, depth_limit=1)


def test_the_merge_appends_the_kept_zerobias_hits_after_the_simulated_ones(two_file_sample):
    picker, _ = two_file_sample
    events = clean_mc(_mc_events(3), CUTS)
    partners, _ = picker.pick("qcd", "mc_a.root", 0, 3)

    merged = add_zerobias(events, partners)

    # CSC: the simulated hit is kept (in time, on the gen jet), then the ZeroBias one
    assert ak.to_list(merged.cscRechits.fromZeroBias) == [[False, True]] * 3
    # DT: the simulated hit is far from everything, so only ZeroBias remains
    assert ak.to_list(merged.dtRecHits.fromZeroBias) == [[True]] * 3
    assert ak.num(merged.cscRechitsZB).tolist() == [1, 1, 1]


def test_a_rechit_field_on_one_side_only_is_refused(two_file_sample):
    picker, _ = two_file_sample
    events = clean_mc(_mc_events(2), CUTS)
    partners, _ = picker.pick("qcd", "mc_a.root", 0, 2)
    partners["dtRecHits"] = ak.with_field(partners.dtRecHits, 1, "llpIdx")

    with pytest.raises(ValueError, match="only ZeroBias \\['llpIdx'\\]"):
        add_zerobias(events, partners)


def test_overlay_apply_names_each_partner(two_file_sample):
    picker, _ = two_file_sample
    config = OverlayConfig(CUTS, ZeroBiasSource("zb_test", "m", data_clean_dr=None))
    overlay = Overlay(config, picker)

    events = overlay.apply(_mc_events(4), "qcd", "mc_a.root", 0, 4)
    expected, _ = picker.pick("qcd", "mc_a.root", 0, 4)

    assert ak.to_list(events.overlay.event) == _event_numbers(expected)
    assert ak.to_list(events.overlay.reused) == [False] * 4
    assert ak.to_list(events.cscRechitsZB.dataCleanKeep) == [[True]] * 4
