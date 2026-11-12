"""Event- and object-level reweighting from ``reweights.yaml``.

Reweighting multiplies extra, analysis-level weights into the event weight
(flattening a spectrum, matching MC kinematics to data, a trigger turn-on).
It complements ``corrections.yaml`` (official correctionlib scale factors);
both go into the same :class:`coffea.analysis_tools.Weights`.

An entry is an **expression** (any NanoEvents expression), or a **binned
map**: ``variable`` looked up in ``edges`` / ``weights`` (1D), or
``variable_x`` / ``variable_y`` with ``edges_x`` / ``edges_y`` (2D).
``file:`` loads the map from another YAML, as ``suep-reweight`` writes it.
The keys (``level``, ``apply_to``, ``samples``, ``clamp``, ``fill_only``)
are described in docs/configuration.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
import yaml

from .config import load_config_file, name_list, weight_concerns
from .fill import compile_expr


def _absolute_map_files(defs: dict, path, axes) -> dict:
    """Resolve a relative ``file:`` against the reweights.yaml that names it.

    Done per file, before the merge, so a map inherited through ``_extends``
    or ``set.yaml`` is still found next to the config that defined it rather
    than looked for in the inheriting set's directory.
    """
    out = {}
    for name, cfg in defs.items():
        if isinstance(cfg, dict) and "file" in cfg and not Path(cfg["file"]).is_absolute():
            cfg = {**cfg, "file": os.path.normpath(Path(path).parent / cfg["file"])}
        out[name] = cfg
    return out


def load_reweight_defs(path: str | Path) -> dict:
    return load_config_file(path, normalize=_absolute_map_files)


def _binned_lookup(vals: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                   clamp: bool) -> np.ndarray:
    """Per-value weight from a 1D binned map (out-of-range/non-finite -> 1)."""
    vals = np.asarray(vals, dtype=np.float64)
    idx = np.searchsorted(edges, vals, side="right") - 1
    n = len(weights)
    finite = np.isfinite(vals)
    if clamp:
        return np.where(finite, weights[np.clip(idx, 0, n - 1)], 1.0)
    inside = finite & (idx >= 0) & (idx < n)
    return np.where(inside, weights[np.clip(idx, 0, n - 1)], 1.0)


def _binned_lookup_2d(vx, vy, edges_x, edges_y, weights, clamp) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    ix = np.searchsorted(edges_x, vx, side="right") - 1
    iy = np.searchsorted(edges_y, vy, side="right") - 1
    nx, ny = weights.shape
    finite = np.isfinite(vx) & np.isfinite(vy)
    if clamp:
        w = weights[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)]
        return np.where(finite, w, 1.0)
    inside = finite & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    w = weights[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)]
    return np.where(inside, w, 1.0)


class Reweighter:
    """One reweighting entry: evaluates per-event or per-object weights."""

    def __init__(self, name: str, cfg: dict, config_dir: str | Path | None = None):
        self.name = name
        self.include_path: Path | None = None

        if "file" in cfg:
            path = Path(cfg["file"])
            if not path.is_absolute() and config_dir is not None:
                path = Path(config_dir) / path
            if not path.exists():
                raise FileNotFoundError(f"reweight '{name}': map file not found: {path}")
            self.include_path = path
            with open(path) as f:
                base = yaml.safe_load(f) or {}
            cfg = {**base, **{k: v for k, v in cfg.items() if k != "file"}}

        self.level = cfg.get("level", "event")
        if self.level not in ("event", "object"):
            raise ValueError(f"reweight '{name}': level must be 'event' or 'object'")
        where = f"reweight '{name}'"
        self.apply_to = name_list(cfg.get("apply_to", "all") or "all", where,
                                  "apply_to", keep=("all",))
        self.samples = name_list(cfg.get("samples"), where, "samples")
        self.fill_only = bool(cfg.get("fill_only", False))
        self.clamp = bool(cfg.get("clamp", False))

        if "expression" in cfg:
            self.kind = "expression"
            self.exprs = [cfg["expression"]]
        elif "variable" in cfg:
            self.kind = "binned"
            self.exprs = [cfg["variable"]]
            self.edges = np.asarray(cfg["edges"], dtype=np.float64)
            self.weights = np.asarray(cfg["weights"], dtype=np.float64)
            if len(self.weights) != len(self.edges) - 1:
                raise ValueError(f"reweight '{name}': need len(weights) == len(edges) - 1 "
                                 f"({len(self.weights)} vs {len(self.edges) - 1})")
        elif "variable_x" in cfg and "variable_y" in cfg:
            self.kind = "binned2d"
            self.exprs = [cfg["variable_x"], cfg["variable_y"]]
            self.edges_x = np.asarray(cfg["edges_x"], dtype=np.float64)
            self.edges_y = np.asarray(cfg["edges_y"], dtype=np.float64)
            self.weights = np.asarray(cfg["weights"], dtype=np.float64)
            expect = (len(self.edges_x) - 1, len(self.edges_y) - 1)
            if self.weights.shape != expect:
                raise ValueError(f"reweight '{name}': weights shape {self.weights.shape} "
                                 f"!= (len(edges_x)-1, len(edges_y)-1) = {expect}")
        else:
            raise ValueError(f"reweight '{name}': need 'expression', 'variable', "
                             "'variable_x'+'variable_y', or 'file'")

    def concerns(self, sample: str, group: str, is_data: bool) -> bool:
        """Whether this reweight is meant for a sample (see config.weight_concerns)."""
        return weight_concerns(self.samples, self.apply_to, sample, group, is_data)

    def applies_to(self, sample: str, group: str, is_data: bool) -> bool:
        """Whether it goes into the event weight of a sample: not if ``fill_only``."""
        return not self.fill_only and self.concerns(sample, group, is_data)

    def evaluate(self, events):
        """Weight array for a chunk: 1D per event, or jagged for level: object."""
        results = [compile_expr(e)(events) for e in self.exprs]

        if self.kind == "expression":
            return results[0]

        if self.level == "object":
            var = ak.fill_none(ak.fill_none(results[0], [], axis=0), np.nan)
            counts = np.asarray(ak.num(var))
            flat = [np.asarray(ak.flatten(var, axis=None))]
            if self.kind == "binned2d":
                var_y = ak.fill_none(ak.fill_none(results[1], [], axis=0), np.nan)
                flat.append(np.asarray(ak.flatten(var_y, axis=None)))
        else:
            flat = [np.asarray(ak.fill_none(r, np.nan)) if isinstance(r, ak.Array)
                    else np.asarray(r) for r in results]
            counts = None

        if self.kind == "binned":
            w = _binned_lookup(flat[0], self.edges, self.weights, self.clamp)
        else:
            w = _binned_lookup_2d(flat[0], flat[1], self.edges_x, self.edges_y,
                                  self.weights, self.clamp)

        if counts is not None:
            return ak.unflatten(w, counts)
        return w

    def event_weight(self, events) -> np.ndarray:
        """Per-event weight (object-level results collapse via product)."""
        w = self.evaluate(events)
        if isinstance(w, ak.Array) and w.ndim > 1:
            w = ak.prod(ak.fill_none(w, 1.0), axis=1)
        w = np.asarray(ak.fill_none(w, 1.0) if isinstance(w, ak.Array) else w,
                       dtype=np.float64)
        return np.where(np.isfinite(w), w, 1.0)


def build_reweighters(rw_defs: dict, config_dir: str | Path | None = None) -> dict[str, Reweighter]:
    return {name: Reweighter(name, cfg, config_dir) for name, cfg in rw_defs.items()}


def apply_reweights(reweighters: dict[str, Reweighter], events, sample: str,
                    group: str, is_data: bool, weights) -> None:
    """Register every applicable reweight into a coffea ``Weights`` object."""
    for name, rw in reweighters.items():
        if rw.applies_to(sample, group, is_data):
            weights.add(name, rw.event_weight(events))


def _unit_weight(events):
    return 1.0


def reweight_weights(reweighters: dict[str, Reweighter], sample: str, group: str,
                     is_data: bool) -> dict:
    """``{name: events -> weights}`` for a histogram's ``weight: "@name"``.

    The map as evaluated for a sample it is meant for, 1 for the others --
    the same rule as for the event weight, ``fill_only`` or not.
    """
    return {name: rw.evaluate if rw.concerns(sample, group, is_data) else _unit_weight
            for name, rw in reweighters.items()}


# ── Map generation (suep-reweight) ────────────────────────────────


def make_reweight_map(
    inputs: list[str],
    hist_name: str,
    num_sample: str,
    den_sample: str,
    out_path: str,
    normalize: bool = True,
    clamp: bool = True,
) -> str:
    """Derive a binned reweight map = <num>/<den> of an existing histogram.

    Reads processed pickles, takes the ratio of the two samples' distributions
    (shape-only when *normalize*), and writes a map YAML whose ``variable`` is
    the histogram's fill expression — ready to reference from reweights.yaml
    via ``file:``.  Bins where the denominator is empty get weight 1.
    """
    from .results import merge_results, resolve_inputs

    data = merge_results(resolve_inputs(inputs))
    if not data or hist_name not in data.get("histograms", {}):
        raise SystemExit(f"histogram '{hist_name}' not found in {inputs}")
    h = data["histograms"][hist_name]
    hdef = data.get("hist_defs", {}).get(hist_name, {})

    cats = list(h.axes["dataset"])
    for s in (num_sample, den_sample):
        if s not in cats:
            raise SystemExit(f"sample '{s}' not in histogram '{hist_name}' "
                             f"(available: {cats})")

    num = h[{"dataset": num_sample}].view().value
    den = h[{"dataset": den_sample}].view().value
    if normalize:
        if num.sum() <= 0 or den.sum() <= 0:
            raise SystemExit(f"cannot normalize: empty numerator or denominator "
                             f"in '{hist_name}'")
        num = num / num.sum()
        den = den / den.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / den, 1.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    axes = [a for a in h.axes if a.name != "dataset"]
    per_object = bool(hdef.get("per_object", False))
    out: dict = {"level": "object" if per_object else "event", "clamp": clamp}

    if len(axes) == 1:
        expr = hdef.get("expression")
        if not expr:
            raise SystemExit(f"histogram '{hist_name}' has no stored fill expression")
        out.update({
            "variable": expr,
            "edges": [float(e) for e in axes[0].edges],
            "weights": [float(w) for w in ratio],
        })
    else:
        expr_x, expr_y = hdef.get("expression_x"), hdef.get("expression_y")
        if not (expr_x and expr_y):
            raise SystemExit(f"histogram '{hist_name}' has no stored fill expressions")
        out.update({
            "variable_x": expr_x,
            "variable_y": expr_y,
            "edges_x": [float(e) for e in axes[0].edges],
            "edges_y": [float(e) for e in axes[1].edges],
            "weights": [[float(w) for w in row] for row in ratio],
        })

    header = (f"# Reweight map generated by suep-reweight\n"
              f"# ratio: {num_sample} / {den_sample} of histogram '{hist_name}'"
              f"{' (shape-only, normalized)' if normalize else ''}\n"
              f"# NOTE: derived within that histogram's selections; the weight is\n"
              f"# applied to all events of the target sample(s).\n")
    with open(out_path, "w") as f:
        f.write(header)
        yaml.safe_dump(out, f, default_flow_style=None, sort_keys=False)
    return out_path
