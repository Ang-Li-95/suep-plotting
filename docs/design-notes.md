# Design notes

Why the code is the way it is: decisions, the reasons behind them, and the
measurements that settled them. The docstrings say what a function does; the
longer story lives here so the code stays short. For how to *use* the
package, see [configuration.md](configuration.md) and [running.md](running.md).

## Where things are

| Module | Job |
|---|---|
| `config.py` | the one YAML reader and its directives |
| `histdefs.py` | `Axis` / `Histogram` / `Selection`, the variable catalog, regions |
| `fill.py` | expressions, selection columns, filling |
| `inputs.py` | samples, input file lists (xrootd), opening one file |
| `configset.py` | loading a whole set (`ConfigSet`) and its custom columns |
| `checks.py` | static checks and the early expression check |
| `processor.py` | the coffea processor and `run_all` (`suep-run`) |
| `results.py` | finding, reading and merging the pickles; restyle, cutflow, ROOT export |
| `figures.py` | CMS style, figure helpers, 1D and 2D plots |
| `derived.py` | `derived_plots.yaml` and its renderers |
| `gallery.py` | `index.html` |
| `plot.py` | `plot_all` (`suep-plot`) |
| `slurm.py`, `status.py` | submission, completion check, resubmission |
| `corrections.py`, `reweights.py`, `jme.py` | scale factors, reweights, JEC/JER |
| `cli.py`, `configtool.py` | the commands |
| `custom/` | the MDS derived columns: `columns.py` (pipeline), `clustering.py`, `llp.py`, `helpers.py`, `params.py` |

Imports run one way: `histdefs` ← `fill` ← `configset` ← `checks` ←
`processor`, and `figures` ← `derived` ← `plot`. `figures.py` and
`configset.py` exist for that reason: the derived plots need the figure
helpers while `plot_all` needs the derived plots, and both the run and the
checks need to load a set.

## Config loading

- **One reader.** Every file of a set goes through `load_config_file`, so a
  directive means the same thing in every file.
- **`_registry` is not `_include`.** Selecting from a pool and merging
  definitions want opposite meanings for an empty value: in a registry a name
  with no value *takes* the pool entry, in a merge `name: null` *deletes* one.
- **`<NAME>` in `_repeat`, not `{NAME}`.** The labels are full of LaTeX
  (`$N_{\mathrm{hits}}$`), so braces were not available as a sigil. A value
  that is exactly one placeholder takes the substitution's type (`hi: <HI>`
  stays an integer, `edges: <EDGES>` a list); embedded in text it is a string.
- **An unknown `_key` is an error.** A misspelled directive (`_extend:`) would
  otherwise be dropped and take everything it should have inherited with it.
- **A `None` is never stored.** In a merge, `name: null` for a key the base
  does not have is dropped: a stored `None` only resurfaces later as an
  unrelated crash.
- **Freshness follows the directives.** `config_sources` lists every file a
  set really reads (`_extends`, `_include`, `_registry` followed). Without it,
  editing a shared fragment left every run that includes it reporting "up to
  date" and plotting stale histograms. `derived_plots.yaml` is deliberately
  not a fill source: it is read at plot time only, so editing an efficiency
  must not force a reprocessing.
- **Regions are the only way to define a histogram** (since 2026-11-10).
  Standalone histograms, the long form and `variants` were removed: three
  ways to write the same thing meant three code paths, and most of the bugs
  found in review sat in the two that the shipped sets did not use.
- **Typed definitions, flat pickles.** The loaders return `Histogram` and
  `Selection` objects; the pickles store `Histogram.to_dict()`, the flat form
  they always stored, so old run directories stay readable.

## Expressions

- Config expressions are evaluated with one namespace passed as `eval`'s
  *globals*. With names split between globals and locals, a lambda or a
  comprehension inside an expression died with `NameError: name 'ak' is not
  defined`: nested scopes resolve names against globals only.
- `custom/` selects objects with the same evaluator (`fill.compile_expr`), so
  a selection means the same thing in a histogram and in the isolation step.

## Selections as columns

Every selection is attached to `events` as a column of its own name.

- **Why.** A histogram can then count what a cut keeps without restating it
  (`ak.sum(events.electron_sel, axis=1)`). The alternative is two copies of
  the same physics, and retuning a working point silently moves only one.
- **Order.** A selection may read another one's column whichever order they
  are written in: what fails is retried until a pass makes no progress, and
  only then is the first remaining failure reported.
- **Missing values** become `False`: `ak.firsts` on an empty event gives
  `None`, which would poison both `ak.sum` and the object filtering.
- **No shadowing.** A selection named after a real branch (`Jet`) is skipped
  with a message; overwriting `events.Jet` would change what every other
  expression means.
- **Fatal when it does not evaluate.** A mask silently missing on one sample
  is a cut silently not applied.
- **Event-level histograms take event-level selections only** (2026-11-10).
  An object mask says which objects pass, not which events; two of them over
  different collections have no event-level reading at all. "Has a passing
  object" is its own selection: `ak.any(events.<sel>, axis=1)`.

## Running and outputs

- **Pickles are written atomically** (temporary file, then rename). A task
  killed mid-write leaves only `*.tmp.<pid>`, which no `*.pkl` glob picks up,
  never a truncated pickle under the real name.
- **Which pickles belong to a run.** `suep-plot` on a `suep-submit` directory
  merges exactly the pickles its tasks wrote in *this* submission. Others in
  the directory would be summed in silently, so they are skipped with a
  warning, and a pickle older than the submission counts as stale.
- **`X.pkl` next to `X.part<k>.pkl` is refused**: the whole sample and a
  shard of it would be summed, counting the shard twice.
- **`restyle` replaces, it does not overlay.** The config's definition
  replaces the pickled one (keeping the recorded `per_object`), so a key
  removed from the config stops applying.
- **`max_files`.** A background sample can be far larger than a shape
  comparison needs: DY is 5,575 files against a signal's 25. The cap takes
  the first N of the sorted list, so a capped run is reproducible and can be
  widened later.
- **One table of derived-plot types** (`DERIVED_TYPES`) is read both to decide
  which histograms a plot needs and to pick its renderer. There used to be
  two copies, which is how a type could be added to one and not the other.

## Jets and MET (`jme.py`)

`correct_jets` is wiring around coffea's `CorrectedJetsFactory` (JEC) and the
official `JERSmear` payload (smearing):

1. resolve the payload tags for the era (`jet_jerc.json.gz`), including the
   `*_DATA` tag whose compound is run-dependent;
2. build the per-jet columns the factory reads through its `name_map`:
   `pt_raw` / `mass_raw` (undoing `rawFactor`), `event_rho`, `run`, and the gen
   pT of jets matched within `dR < 0.2` and 3 sigma, which selects the hybrid
   smearing method per jet;
3. pick the requested variation (`JES_Total.up/down`, `JER.up/down`);
4. rebuild the Type-1 MET from the **raw** MET: subtract the vector sum of
   `pT_L2L3Res - pT_L1` over `Jet` and `CorrT1METJet`, muon-subtracted, for
   jets with corrected pT > 15 GeV, `|eta| < 5.2` and EM fraction < 0.9;
5. re-sort the jets by the new pT and drop the helper columns.

**Payloads.** `payload_path` looks in `$CORRECTIONLIB_DATA` first, then the
era's CAT campaign directory, then cvmfs jsonpog-integration. The campaign
directory matters: jsonpog's `2024_Summer24` still carries only
`Summer24Prompt24_V1`, while the 2024 recommendation is `V5` (V3→V4 bumped the
tag without updating the L2L3Residual payloads; V5 is the fixed one). The
`JERSmear` payload is era-independent and shipped next to the module.

**Muon subtraction is asymmetric on purpose.** The jets handed back are the
whole jet, muon content included: that is what the JEC was derived for, and
what jet pT, HT and the jet selections should use. Only the MET sum uses
`1 - muonSubtrFactor`, because MET already counts the muon directly and
correcting its momentum as hadronic energy would double count it.

**Closure of the MET rebuild** (2000 signal events). Feeding the *production*
jet pT through the same machinery reproduces the production Type-1 term to
0.001 GeV (14.165 vs 14.166 GeV mean magnitude), which is what validates the
recipe. Rebuilding with the current calibration lands about 6 GeV from the
stored PuppiMET, because the JEC has moved since the samples were produced:
the stored jets carry a mean correction factor of 1.58 against 1.25 for
`Summer24Prompt24_V5`. That difference is the physics, and the reason to
rebuild rather than keep the stored value. Two fidelity notes: the muon
subtraction is applied here (mean `muonSubtrFactor` about 0.012, a ~1 %
effect) while the production term closes without it; and for PUPPI the L1
factor is exactly 1.0, so `pT_L2L3Res - pT_L1` reduces to the correction
above raw.

## Clustering (`custom/clustering.py`)

- **Match RPC, do not merge it** (the default `rpc_mode: match`). RPC rechit
  times are flat in BX in data: only about 19 % of the RPC rechits of a
  ZeroBias event sit at BX 0. Merged into the clustering, RPC noise can push
  a background cluster over `min_samples`; matched afterwards, it can only
  annotate a cluster the CSC or DT rechits already made. The cluster
  variables then keep the values, and the signal region the definition, of
  the standard MDS analysis. Both modes attach the same RPC fields, so one
  config set plots either.
- **Id offsets in a merge.** A DT layer key and an RPC layer key are both
  small integers, and a DT and an RPC chamber code can be identical, so the
  RPC component's ids are offset. The offset is keyed on the system, not on
  the order of the components: RPC is never the primary system, so CSC and DT
  keep the ids they always had and the shift means "RPC" in both merged
  collections.
- **RPC rechit time is not used**, only the BX: see
  [derived-columns.md](derived-columns.md).
- **Memory grows as N² per event**, from the precomputed distance matrix that
  handles the phi wrap-around: see [known-issues.md](known-issues.md).
- **One function per group of fields** (`_position`, `_stations`,
  `_hits_per_unit`, `_first_hit`, `_truth_match`, `_hit_time`,
  `_rpc_content`). The loop that used to compute all of them inline was 170
  lines.

## LLP rechit spread (`custom/llp.py`)

- `_dr_kernel` is compiled with numba because the groups are tiny (about 3
  rechits per LLP): in numpy the loop was about 17 times slower, spending
  almost all of it on per-call dispatch. The results match the numpy version
  to the last ulp (numpy sums pairwise, the kernel in a plain loop).
- `pair_max_hits` is passed to the kernel as an argument: numba would freeze
  a module-level constant into the cached machine code at the first compile.

## Checking a change

A change to the framework must not move a histogram. The recipe used for the
2026-11-10 restructuring: dump the definitions of every set
(`scripts/dump_config.py`), fill one signal and one data file before and
after, and compare bin by bin; for `custom/clustering.py`, compare every
field of every cluster. See also "Checking a change to the framework" in
[running.md](running.md).
