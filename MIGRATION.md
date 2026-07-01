# Migration to coffea (v0.1 → v0.2)

The processing layer was rebased on **coffea** (NanoEvents + `ProcessorABC` +
`Runner`), and corrections now use **coffea's own tooling**
(`correctionlib_wrapper` + `analysis_tools.Weights`). The plotting layer
(`plot.py`, `derived_plots.yaml`) is unchanged. Validated end-to-end in
`conda activate mds` (coffea 2026.5.0) against the MDSNano file in `samples.yaml`.

## What changed

- **Reading:** raw `uproot.iterate` + `ev['Branch']` dicts → coffea
  `NanoEventsFactory`/`NanoAODSchema` driven by `processor.Runner`. Expressions
  are now NanoEvents object syntax (`events.Jet.pt`, `ak.num(events.Muon)`), which
  unlocks Lorentz-vector behaviours, `delta_r`, `nearest`, etc.
- **Processing:** `SuepProcessor(ProcessorABC)` fills the YAML-defined histograms;
  `run_all()` runs it per sample (keeps the per-sample Slurm-array model).
  `--workers > 1` switches to coffea `FuturesExecutor`.
- **Corrections:** `corrections.py` rewritten around `correctionlib_wrapper` and
  `Weights`. `event_weight` and `object_sf` both supported; `object_sf` flattens
  jagged inputs and passes scalar/string systematics through, so the correctionlib
  string axis works (this fixes the old byte-code corruption bug). Per-event SF =
  `ak.prod(ak.unflatten(sf, counts), axis=1)`.
- **scipy shim:** `src/suep_plot/_compat.py` restores the `_lazywhere` helper that
  newer scipy removed, so coffea's `lookup_tools` (hence `correctionlib_wrapper`)
  imports. Applied from `suep_plot/__init__.py` before coffea loads; no change to
  the shared `mds` env.
- **Configs:** `histograms.yaml`, `selections.yaml`, `corrections.yaml` ported to
  NanoEvents syntax. `custom/columns.py` hook is now `derive(events) -> events`.

## Review fixes folded in (from the prior audit)

- **CLI dispatcher** added — `python -m suep_plot.cli {run,plot,submit}` now works
  (was a silent no-op).
- **Build backend** fixed (`setuptools.build_meta`); `pip install -e .` works.
- **Slurm** scripts now `export PYTHONPATH=<repo>/src` — tasks import the package.
- **scipy** added to dependencies; unused `dask`/`rich` removed; `coffea` promoted
  to a core dependency.
- **Missing-branch feedback:** a one-time expression-validation pass warns on
  typo'd fields instead of silently producing empty histograms.
- **Docs:** README rewritten for the coffea design; histogram count corrected
  (25), object-level selections documented, unused per-sample `lumi` removed.

## Validation performed

- `python -m suep_plot.cli run` → 25 histograms filled (parity: 40 jets, 20 muons,
  11732 SUEP gen particles, etc.); `plot` → 30 PNG/PDF incl. 5 derived plots.
- Corrections: synthetic `event_weight` (×1.05) and per-muon `object_sf` (×1.10,
  string systematic) applied through the processor; isolated check confirmed the
  per-event weight equals `1.10^N_muon` exactly.
- Slurm `--dry-run` emits `job.sh`/`merge_and_plot.sh` with the `PYTHONPATH`
  export; build backend and `pyproject` parse cleanly.

## Not yet done (optional follow-ups)

- **xs × lumi normalization** — `sumw` (Σ genWeight) and `nevents` are now stored
  per sample in the pickle, so scaling each MC sample by
  `xs * lumi * 1000 / sumw` at plot time is a small `plot.py` addition.
- **Systematic variations** — `Weights` supports `add(name, nominal, up, down)`;
  the config schema could grow an optional up/down input set.
- **JEC/JER** — `coffea.jetmet_tools` now imports (via the shim) if you later want
  jet energy corrections applied to `events.Jet` before filling.
