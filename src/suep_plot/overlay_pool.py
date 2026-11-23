"""Which ZeroBias event each simulated event is overlaid with.

The pairing is decided before any event is processed and frozen in the run
directory, so that independent Slurm tasks, chunks and resubmissions never
use one ZeroBias event twice within a sample, and a run can be repeated:

``<run>/overlay/pool.json``
    the pool: its files (sorted), the run's seed, the lumi mask
``<run>/overlay/<sample>.json``
    the sample's MC files and, for each, its block of pool files

How a pair is made (docs/design-notes.md, "The ZeroBias overlay"):

1. Per sample, the P pool files are shuffled (seed: run seed and sample name)
   and cut into G disjoint blocks of k = P // G files, one per MC file.  Two
   MC files of a sample therefore never share a ZeroBias file.
2. MC file g reads its block as a stream s_g: the block's files in their
   shuffled order and, inside each file, the entries the lumi mask certifies,
   shuffled by a seeded hash.  Every certified event of the block is in s_g
   once.
3. MC entry j of file g is paired with s_g[j].  j is the entry number in the
   MC file, so chunk size, task order and resubmission cannot change a pair.
   Only if a file has more events than its block holds does s_g wrap around;
   those events are flagged as reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import awkward as ak
import numpy as np

PLAN_DIR_NAME = "overlay"
POOL_FILE = "pool.json"
# Entries read from a ZeroBias file at a time: the shuffled stream needs
# events from all over a file, and reading it in steps bounds the memory.
READ_STEP = 5000
EVENT_ID_BRANCHES = ("run", "luminosityBlock", "event")


@dataclass(frozen=True)
class PoolPlan:
    """``pool.json``: the ZeroBias files of a run and how they are shuffled."""

    name: str                  # the datasets.yaml entry
    lumi_mask: str             # as overlay.yaml gives it
    lumi_mask_sha256: str
    seed: int                  # drawn from the clock when the run's plan is made
    files: tuple[str, ...]


@dataclass(frozen=True)
class SamplePlan:
    """``<sample>.json``: each MC file of a sample and its block of pool files."""

    sample: str
    mc_files: tuple[str, ...]
    blocks: tuple[tuple[int, ...], ...]   # pool indices, one block per MC file

    def block_of(self, mc_file: str) -> tuple[int, ...]:
        try:
            return self.blocks[self.mc_files.index(mc_file)]
        except ValueError:
            raise ValueError(
                f"overlay: {mc_file} is not in the frozen pairing plan of sample "
                f"'{self.sample}'.  The plan lists the files the run was set up "
                "with; process this file in a new output directory.") from None


# ── making and reading the plans ──────────────────────────────────


def plan_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / PLAN_DIR_NAME


def prepare_plans(run_dir, pool_name: str, lumi_mask: str,
                  sample_files: dict[str, list[str]]) -> None:
    """Make the run's pool plan and one plan per sample, unless they exist.

    An existing plan is reused as it is (that is what keeps a resubmission on
    the same pairs); it must have been made for the same pool and lumi mask.
    *sample_files* maps each simulated sample to its full, sorted file list.
    """
    directory = plan_dir(run_dir)
    pool = _ensure_pool(directory, pool_name, lumi_mask)
    for sample, files in sample_files.items():
        if not (directory / f"{sample}.json").exists():
            plan = build_sample_plan(sample, files, pool)
            _write_once(directory / f"{sample}.json", _to_json(plan))


def load_pool(directory: Path) -> PoolPlan:
    path = Path(directory) / POOL_FILE
    if not path.exists():
        raise SystemExit(f"ERROR: no overlay plan in {directory}: suep-run and "
                         "suep-submit make it before processing")
    raw = json.loads(path.read_text())
    return PoolPlan(**{**raw, "files": tuple(raw["files"])})


def load_sample_plan(directory: Path, sample: str) -> SamplePlan:
    path = Path(directory) / f"{sample}.json"
    if not path.exists():
        raise SystemExit(f"ERROR: no overlay plan for sample '{sample}' in "
                         f"{directory}: suep-run and suep-submit make it before "
                         "processing")
    raw = json.loads(path.read_text())
    return SamplePlan(raw["sample"], tuple(raw["mc_files"]),
                      tuple(tuple(block) for block in raw["blocks"]))


def build_sample_plan(sample: str, mc_files: list[str], pool: PoolPlan) -> SamplePlan:
    """One disjoint block of k = P // G pool files per MC file (see module doc)."""
    n_pool, n_mc = len(pool.files), len(mc_files)
    files_per_block = n_pool // n_mc if n_mc else 0
    if files_per_block == 0:
        raise SystemExit(f"ERROR: overlay: sample '{sample}' has {n_mc} files but "
                         f"the ZeroBias pool only {n_pool}; every MC file needs "
                         "ZeroBias files of its own")
    rng = np.random.default_rng(_sample_seed(pool.seed, sample))
    order = rng.permutation(n_pool)
    blocks = tuple(tuple(int(i) for i in order[g * files_per_block:
                                               (g + 1) * files_per_block])
                   for g in range(n_mc))
    return SamplePlan(sample, tuple(mc_files), blocks)


def pool_definition(name: str) -> dict:
    """The pool's entry in datasets.yaml."""
    from .config import REPO_ROOT, load_config_file

    registry = load_config_file(REPO_ROOT / "datasets.yaml")
    if name not in registry:
        raise SystemExit(f"ERROR: overlay: ZeroBias pool '{name}' is not in "
                         "datasets.yaml")
    return registry[name]


def _ensure_pool(directory: Path, pool_name: str, lumi_mask: str) -> PoolPlan:
    from .inputs import lumi_mask_path, resolve_sample_files

    mask_sha = _sha256(lumi_mask_path(lumi_mask))
    if (directory / POOL_FILE).exists():
        pool = load_pool(directory)
        if (pool.name, pool.lumi_mask_sha256) != (pool_name, mask_sha):
            raise SystemExit(
                f"ERROR: the overlay plan in {directory} was made for pool "
                f"'{pool.name}' with lumi mask {pool.lumi_mask}; overlay.yaml now "
                f"says '{pool_name}' with {lumi_mask}.  Use a new output directory.")
        return pool

    print(f"Overlay: listing the ZeroBias pool '{pool_name}' ...")
    files = resolve_sample_files(pool_definition(pool_name))
    if not files:
        raise SystemExit(f"ERROR: overlay: the ZeroBias pool '{pool_name}' has no files")
    pool = PoolPlan(pool_name, str(lumi_mask), mask_sha, time.time_ns(), tuple(files))
    raw = _write_once(directory / POOL_FILE, asdict(pool))
    print(f"Overlay: {len(files)} ZeroBias files, seed {raw['seed']}, "
          f"plans in {directory}")
    return PoolPlan(**{**raw, "files": tuple(raw["files"])})


def _write_once(path: Path, payload: dict) -> dict:
    """Write *path* unless it exists; return what is on disk.

    Hard-linking a complete temporary file is atomic and fails when the
    target exists, so two runs starting together agree on one plan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    try:
        os.link(tmp, path)
    except FileExistsError:
        pass
    finally:
        tmp.unlink()
    return json.loads(path.read_text())


def _to_json(plan: SamplePlan) -> dict:
    return {"sample": plan.sample, "mc_files": list(plan.mc_files),
            "blocks": [list(block) for block in plan.blocks]}


def _sample_seed(seed: int, sample: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── picking the partners of a chunk ───────────────────────────────


class ZeroBiasPicker:
    """Reads the ZeroBias partners of a chunk of simulated events.

    Loads the run's plans lazily, so the object pickled to every worker is
    small; each worker caches the plans, the lumi mask and the certified,
    shuffled entry list of every ZeroBias file it has opened.
    """

    def __init__(self, directory: Path | None, collections: tuple[str, ...]):
        self.directory = None if directory is None else Path(directory)
        # The ZeroBias collections to read, besides run/lumi/event.
        self.collections = tuple(collections)
        self._pool: PoolPlan | None = None
        self._plans: dict[str, SamplePlan] = {}
        self._certified: dict[int, np.ndarray] = {}
        self._lumi_mask = None

    @classmethod
    def in_memory(cls, pool: PoolPlan, plans: dict[str, SamplePlan],
                  collections: tuple[str, ...]) -> "ZeroBiasPicker":
        """A picker on plans that live nowhere on disk (``suep-config check``)."""
        picker = cls(None, collections)
        picker._pool, picker._plans = pool, dict(plans)
        return picker

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.directory is not None:   # the workers reload from disk
            state.update(_pool=None, _plans={}, _certified={}, _lumi_mask=None)
        state["_lumi_mask"] = None       # coffea's LumiMask is rebuilt per worker
        return state

    def pick(self, sample: str, mc_file: str, entry_start: int, entry_stop: int):
        """ZeroBias partners of MC entries [entry_start, entry_stop) of *mc_file*.

        Returns the partner events (``run``, ``luminosityBlock``, ``event`` and
        the collections) in MC entry order, and per event whether its partner
        is reused because the block ran out.
        """
        block = self._sample_plan(sample).block_of(mc_file)
        counts = []          # certified events per block file, as far as needed
        for pool_index in block:
            if sum(counts) >= entry_stop:
                break
            counts.append(len(self._certified_entries(pool_index)))
        available = sum(counts)
        if available == 0:
            raise ValueError(f"overlay: none of the {len(block)} ZeroBias files "
                             f"assigned to {mc_file} has a certified event")

        positions = np.arange(entry_start, entry_stop)
        reused = positions >= available
        positions = positions % available
        ends = np.cumsum(counts)
        slot = np.searchsorted(ends, positions, side="right")
        offset = positions - (ends[slot] - np.asarray(counts)[slot])

        pieces, rows = [], []
        for file_slot in np.unique(slot):
            in_file = np.flatnonzero(slot == file_slot)
            pool_index = block[file_slot]
            entries = self._certified_entries(pool_index)[offset[in_file]]
            pieces.append(self._read(pool_index, entries))
            rows.append(in_file)
        partners = ak.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return partners[np.argsort(np.concatenate(rows))], reused

    # -- cached state of one worker --

    def _pool_plan(self) -> PoolPlan:
        if self._pool is None:
            self._pool = load_pool(self.directory)
        return self._pool

    def _sample_plan(self, sample: str) -> SamplePlan:
        if sample not in self._plans:
            if self.directory is None:
                raise ValueError(f"overlay: no pairing plan for sample '{sample}'")
            self._plans[sample] = load_sample_plan(self.directory, sample)
        return self._plans[sample]

    def _mask(self):
        if self._lumi_mask is None:
            from coffea.lumi_tools import LumiMask

            from .inputs import lumi_mask_path

            self._lumi_mask = LumiMask(str(lumi_mask_path(self._pool_plan().lumi_mask)))
        return self._lumi_mask

    def _certified_entries(self, pool_index: int) -> np.ndarray:
        """The entries of one pool file the lumi mask certifies, in stream order."""
        if pool_index not in self._certified:
            import uproot

            pool = self._pool_plan()
            with uproot.open(pool.files[pool_index]) as root_file:
                ids = root_file["Events"].arrays(["run", "luminosityBlock"],
                                                 library="np")
            certified = np.flatnonzero(self._mask()(
                ids["run"].astype(np.uint32), ids["luminosityBlock"].astype(np.uint32)))
            self._certified[pool_index] = shuffle_entries(certified, pool.seed, pool_index)
        return self._certified[pool_index]

    def _read(self, pool_index: int, entries: np.ndarray):
        """The events at *entries* (any order, repeats allowed) of one pool file."""
        import uproot

        path = self._pool_plan().files[pool_index]
        branches = [*EVENT_ID_BRANCHES, *(f"{c}_*" for c in self.collections)]
        pieces, rows = [], []
        with uproot.open(path) as root_file:
            tree = root_file["Events"]
            for start in range(0, tree.num_entries, READ_STEP):
                stop = min(start + READ_STEP, tree.num_entries)
                in_step = np.flatnonzero((entries >= start) & (entries < stop))
                if not len(in_step):
                    continue
                arrays = tree.arrays(filter_name=branches, entry_start=start,
                                     entry_stop=stop)
                pieces.append(arrays[entries[in_step] - start])
                rows.append(in_step)
        flat = ak.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        flat = flat[np.argsort(np.concatenate(rows))]
        return _as_collections(flat, self.collections)


def _as_collections(flat, collections) -> ak.Array:
    """NanoAOD's flat branches (``Jet_pt``) -> one record per object (``Jet.pt``)."""
    out = {name: flat[name] for name in EVENT_ID_BRANCHES}
    for collection in collections:
        prefix = collection + "_"
        fields = [f for f in ak.fields(flat) if f.startswith(prefix)]
        if not fields:
            raise ValueError(f"overlay: the ZeroBias files have no '{collection}' "
                             "collection")
        out[collection] = ak.zip({f[len(prefix):]: flat[f] for f in fields})
    return ak.zip(out, depth_limit=1)


# ── the in-file shuffle ───────────────────────────────────────────


def shuffle_entries(entries: np.ndarray, seed: int, pool_index: int) -> np.ndarray:
    """*entries* reordered by a hash of (seed, pool file, entry).

    A fixed integer hash, not a numpy random stream, so the order depends on
    the seed alone and not on the numpy version.
    """
    salt = _splitmix64(np.array([seed], dtype=np.uint64) ^ np.uint64(pool_index))
    keys = _splitmix64(salt + np.asarray(entries).astype(np.uint64))
    return np.asarray(entries)[np.argsort(keys, kind="stable")]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer, element-wise (uint64 arithmetic wraps)."""
    z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))
