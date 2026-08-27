# Known issues

## Clustering memory scales as N² per event

`custom/clustering.py:_cluster_hits` clusters each event with a **precomputed
N×N distance matrix** (`DBSCAN(metric="precomputed")`, around line 386). The
precomputed metric is there to handle phi wrap-around, but it makes memory grow
as the square of the rechit count *in a single event*.

Collision data is unaffected — p99.9 is about 400 rechits per system, so a few
MB. Cosmics running is not: detector-wide bursts occur, and file 75 of
`cosmics_2024C` contains a DT event with **65,766 rechits**, needing **34.6 GB**
for the matrix alone.

Measured consequence (2026-08-25, the 118-file cosmics run): 27 array tasks
died with `OUT_OF_MEMORY` at 8 GB, 3 more at 32 GB, and they only completed at
120 GB.

> **Reducing `--chunk-size` does not help.** The matrix is per event, not per
> chunk. The only lever today is `--mem`.

Measure the occupancy of a sample with `scratch/cosmics_occupancy.py`.

### Routes to a fix

- **Embed phi** so a tree-based metric works: cluster on
  `(eta, cos(phi), sin(phi))` with the euclidean metric, which sklearn indexes
  in O(N log N) memory. The chord-vs-arc difference is ~0.7 % at `eps = 0.4`,
  so cluster membership changes at the boundary — this is **not** bit-identical
  to current output and the shift has to be quantified before adopting it.
- **A `cluster_max_hits` guard** in `custom/params.py:PARAM_SPEC`, skipping or
  downsampling events above a threshold. There is precedent in `pair_max_hits`
  (`custom/llp.py`). Whatever it drops must be counted and reported, never
  silent.

Either route changes physics output for every sample, so it is a deliberate
decision rather than a quiet fix. Verify with the usual recipe: an empty
`scripts/dump_config.py` diff for config equivalence, plus a bit-by-bit
histogram comparison of one signal file before and after.
