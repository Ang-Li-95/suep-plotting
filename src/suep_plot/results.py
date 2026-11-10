"""The pickles a run wrote: finding, reading and merging them.

``suep-plot`` and the comparison scripts start here: which pickles belong to
a run directory, the per-sample histograms summed into one result, and what
is done to that result before drawing (cross-section scaling, restyling from
the current config) or instead of it (the cutflow tables, the ROOT export).
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import hist


def load_results(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, AttributeError, ValueError) as e:
        raise SystemExit(
            f"ERROR: cannot read {path}: {type(e).__name__}: {e}\n"
            "  A truncated pickle comes from a task killed while writing; "
            "'suep-status -o <run dir> --resubmit' reruns it.") from None


_PART = re.compile(r"^(?P<sample>.+)\.part[^.]+\.pkl$")


def sample_of(pkl: str) -> tuple[str, bool]:
    """(sample, is_part) for ``<sample>.pkl`` / ``<sample>.part<k>.pkl``."""
    name = Path(pkl).name
    m = _PART.match(name)
    return (m.group("sample"), True) if m else (name[:-len(".pkl")], False)


def _run_dir_pickles(run: Path) -> list[str]:
    """The pickles a ``suep-submit`` run's tasks wrote in *this* submission.

    Other ``*.pkl`` files in the directory -- left over from an earlier run, or
    belonging to no task -- would be summed in silently, so they are skipped
    with a warning; tasks that have not written their pickle yet are reported,
    because their samples come out undercounted.
    """
    from .status import classify

    by_state, extra = classify(str(run), verify=False)
    n_tasks = sum(len(v) for v in by_state.values())
    unfinished = []
    if by_state["missing"]:
        unfinished.append(f"{len(by_state['missing'])} missing")
    if by_state["stale"]:
        unfinished.append(f"{len(by_state['stale'])} older than the submission "
                          "(skipped)")
    if unfinished:
        samples = sorted({t.sample for t in by_state["missing"] + by_state["stale"]})
        print(f"WARNING: {run}: {len(by_state['done'])}/{n_tasks} task pickles "
              f"usable ({', '.join(unfinished)}); these samples are undercounted: "
              f"{', '.join(samples)}.  See 'suep-status -o {run}'.")
    if extra:
        print(f"WARNING: {run}: skipping {len(extra)} pickle(s) that no task of "
              f"this run wrote: {', '.join(p.name for p in extra[:5])}"
              f"{' ...' if len(extra) > 5 else ''}")
    return [str(run / t.pkl_name) for t in sorted(by_state["done"],
                                                  key=lambda t: t.index)]


def resolve_inputs(inputs: str | list[str]) -> list[str]:
    """Turn a mix of pickle files and run directories into the pickles to merge.

    A directory written by ``suep-submit`` contributes exactly its tasks'
    pickles (see :func:`_run_dir_pickles`); any other directory contributes
    every ``*.pkl`` in it.  A path that does not exist is an error, a pickle
    named twice is read once, and a directory holding both ``X.pkl`` and
    ``X.part<k>.pkl`` is refused -- the whole sample and a shard of it would
    be summed, counting the shard's events twice.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    out = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            if (p / "slurm" / "task_list.txt").exists():
                out.extend(_run_dir_pickles(p))
            else:
                out.extend(sorted(str(f) for f in p.glob("*.pkl")))
        elif p.is_file():
            out.append(str(p))
        else:
            raise SystemExit(f"ERROR: input not found: {entry}")

    seen, unique = set(), []
    for f in out:
        key = str(Path(f).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    kinds: dict[tuple[str, str], set] = {}
    for f in unique:
        sample, is_part = sample_of(f)
        kinds.setdefault((str(Path(f).resolve().parent), sample), set()).add(is_part)
    both = sorted(f"{d}/{s}" for (d, s), k in kinds.items() if k == {True, False})
    if both:
        raise SystemExit(
            "ERROR: both a whole-sample pickle and part pickles of the same "
            f"sample would be merged (double counting): {', '.join(both)}.\n"
            "  Move the one that does not belong to this run out of the "
            "directory.")
    return unique


def merge_results(paths: list[str]) -> dict:
    """Merge multiple pickle files into one combined result.

    Histograms are summed, so the same sample may be spread over several
    pickles (e.g. ``<sample>.part<k>.pkl`` from a file-split Slurm run);
    its ``sumw``/``nevents``/cutflow counts are summed accordingly.
    """
    merged = None
    for p in paths:
        data = load_results(p)
        if merged is None:
            merged = data
            continue
        for name, h in data["histograms"].items():
            if name in merged["histograms"]:
                merged["histograms"][name] += h
            else:
                merged["histograms"][name] = h
        merged["samples"].update(data.get("samples", {}))
        for key in ("sumw", "nevents"):
            tgt = merged.setdefault(key, {})
            for s, v in data.get(key, {}).items():
                tgt[s] = tgt.get(s, 0) + v
        cf = merged.setdefault("cutflow", {})
        for s, flows in data.get("cutflow", {}).items():
            if s not in cf:
                cf[s] = flows
                continue
            for sel, counts in flows.items():
                if sel in cf[s]:
                    cf[s][sel] = {"raw": cf[s][sel]["raw"] + counts["raw"],
                                  "wtd": cf[s][sel]["wtd"] + counts["wtd"]}
                else:
                    cf[s][sel] = counts
        defs = merged.setdefault("hist_defs", {})
        for name, cfg in data.get("hist_defs", {}).items():
            if name not in defs:
                defs[name] = cfg
                continue
            # per_object is recorded by the fill; a pickle whose sample never
            # filled the histogram has none, so take it from any that has.
            have, new = defs[name].get("per_object"), cfg.get("per_object")
            if have is None and new is not None:
                defs[name] = {**defs[name], "per_object": new}
            elif None not in (have, new) and have != new:
                print(f"  WARNING: histogram '{name}' is per-object in some "
                      "pickles and per-event in others")
    return merged


# ── before drawing ────────────────────────────────────────────────


def apply_xs_scaling(histograms: dict, sample_defs: dict, sumw: dict, lumi: float) -> None:
    """Scale each MC sample to xs [pb] x lumi [/fb] using its summed genWeight.

    weight = xs * lumi * 1000 / sumw.  Data samples and MC samples without
    ``xs`` or a recorded ``sumw`` are left untouched (a warning is printed).
    """
    placeholders = sorted(s for s, cfg in sample_defs.items()
                          if not cfg.get("is_data", False) and cfg.get("xs") == 1.0)
    if placeholders:
        print(f"  WARNING: xs = 1.0 for {', '.join(placeholders)} -- a placeholder, "
              "not a cross section: --lumi scales these samples by a meaningless "
              "factor (and distorts efficiency intervals). Drop --lumi, or set the "
              "real xs in datasets.yaml.")
    factors = {}
    for s, cfg in sample_defs.items():
        if cfg.get("is_data", False):
            continue
        xs = cfg.get("xs")
        sw = sumw.get(s, 0.0)
        if xs is None or not sw:
            missing = "xs" if xs is None else "sumw"
            print(f"  WARNING: MC sample '{s}' has no {missing}; left unscaled")
            continue
        factors[s] = float(xs) * lumi * 1000.0 / sw
    if not factors:
        return

    print("Normalizing MC to xs x lumi (weight = xs * lumi * 1000 / sumw):")
    for s, f in factors.items():
        print(f"  {s}: x {f:.4g}")

    for h in histograms.values():
        cats = list(h.axes["dataset"])
        view = h.view(flow=True)
        for s, f in factors.items():
            if s in cats:
                idx = cats.index(s)
                view["value"][idx] *= f
                view["variance"][idx] *= f * f


def restyle(hist_defs: dict, sample_defs: dict, config_dir) -> None:
    """Re-read plot-time styling from a config set, in place.

    The pickles carry the histogram and sample definitions as they were at
    fill time; this replaces them with the config set's current ones, so a
    label, colour, rebin or log-axis change needs no reprocessing.  (What was filled
    -- the axes, the expression -- is the histogram itself and cannot change.)
    """
    from .config import load_config_file
    from .histdefs import load_histogram_defs

    # The config's definition replaces the pickled one rather than being laid
    # over it, so a key *removed* from the config (a rebin, a log axis, a
    # colour) stops applying too.  Only ``per_object`` is kept from the
    # pickle: the fill recorded it, the config normally does not state it.
    for name, h in load_histogram_defs(Path(config_dir) / "histograms.yaml").items():
        recorded = {k: v for k, v in hist_defs.get(name, {}).items()
                    if k == "per_object"}
        hist_defs[name] = {**recorded, **h.to_dict()}
    for name, cfg in (load_config_file(Path(config_dir) / "samples.yaml") or {}).items():
        if isinstance(cfg, dict):
            sample_defs[name] = dict(cfg)


# ── other outputs ─────────────────────────────────────────────────


def write_cutflow(cutflow: dict, output_dir: str):
    """Write per-sample cutflow tables (raw and weighted counts) as txt + csv.

    ``cutflow`` maps sample -> selection -> {"raw": int, "wtd": float}.
    Counts are independent per selection, not sequential.
    """
    if not cutflow:
        return
    os.makedirs(output_dir, exist_ok=True)

    sel_names: list[str] = []
    for flows in cutflow.values():
        for sel in flows:
            if sel not in sel_names:
                sel_names.append(sel)

    txt_path = os.path.join(output_dir, "cutflow.txt")
    csv_path = os.path.join(output_dir, "cutflow.csv")

    with open(txt_path, "w") as f:
        f.write("Cutflow (independent counts per selection; 'total' = all processed events)\n")
        for sample, flows in cutflow.items():
            f.write(f"\n{sample}\n")
            f.write(f"  {'selection':<28}{'raw':>12}{'weighted':>16}\n")
            for sel in sel_names:
                if sel not in flows:
                    continue
                c = flows[sel]
                f.write(f"  {sel:<28}{c['raw']:>12,}{c['wtd']:>16.4g}\n")

    with open(csv_path, "w") as f:
        f.write("sample,selection,raw,weighted\n")
        for sample, flows in cutflow.items():
            for sel in sel_names:
                if sel in flows:
                    c = flows[sel]
                    f.write(f"{sample},{sel},{c['raw']},{c['wtd']:.6g}\n")

    print(f"Wrote cutflow tables: {txt_path}, {csv_path}")


def export_root(histograms: dict[str, hist.Hist], path: str):
    """Write every histogram x dataset slice to a ROOT file (TH1D/TH2D).

    Layout: one directory per histogram, one key per sample
    (``<hist>/<sample>``), ready for combine datacards or ROOT-based tooling.
    """
    import uproot

    n = 0
    with uproot.recreate(path) as f:
        for name, h in histograms.items():
            for s in h.axes["dataset"]:
                f[f"{name}/{s}"] = h[{"dataset": s}]
                n += 1
    print(f"Wrote {n} histogram(s) to {path}")
