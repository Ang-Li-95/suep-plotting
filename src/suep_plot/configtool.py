"""``suep-config``: read and check a config set without running it.

``show``   what a set really fills, after _extends / _repeat / regions /
           variants are applied, and which file each definition comes from;
``check``  everything that can go wrong before a run: references, columns.yaml,
           and every expression evaluated on one file of each sample;
``fields`` what an expression can use: the collections and fields of a
           sample after derive(), optionally with their value ranges.
"""

from __future__ import annotations

import re
from pathlib import Path


# ── show ──────────────────────────────────────────────────────────


def _binning(cfg: dict, s: str = "") -> str:
    if f"edges{s}" in cfg:
        edges = cfg[f"edges{s}"]
        return f"edges({len(edges) - 1}) [{edges[0]}, {edges[-1]}]"
    return f"{cfg.get(f'bins{s}')} [{cfg.get(f'lo{s}')}, {cfg.get(f'hi{s}')}]"


def _short(text, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + "…"


def _rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _table(rows: list[tuple], widths: tuple) -> None:
    for row in rows:
        print("  " + "  ".join(_short(c, w).ljust(w) for c, w in zip(row, widths)).rstrip())


def show(config_dir: str, grep: str | None = None, sections=None) -> int:
    from .config import load_config_file, read_set_file
    from .histograms import (is_2d, load_histogram_defs, load_selection_defs,
                             load_variables, normalize_hist_file,
                             normalize_selection_file)
    from .plot import load_derived_plot_defs

    cdir = Path(config_dir)
    base = cdir.resolve().parent
    sections = set(sections or ("regions", "histograms", "selections", "derived"))
    pattern = re.compile(grep) if grep else None

    def keep(name: str) -> bool:
        return pattern is None or bool(pattern.search(name))

    h_origins: dict = {}
    raw = load_config_file(cdir / "histograms.yaml", with_fragments=True,
                           normalize=normalize_hist_file, origins=h_origins)
    hists = load_histogram_defs(cdir / "histograms.yaml")
    s_origins: dict = {}
    load_config_file(cdir / "selections.yaml", normalize=normalize_selection_file,
                     origins=s_origins)
    sels = load_selection_defs(cdir / "selections.yaml")
    variables = load_variables(cdir)
    derived = load_derived_plot_defs(str(cdir / "derived_plots.yaml"))

    meta = read_set_file(cdir)
    print(f"{cdir}" + (f"  -- {meta['description']}" if meta.get("description") else ""))
    if meta.get("extends"):
        print(f"  set.yaml: extends {meta['extends']}")
    print(f"  {len(hists)} histograms, {len(sels)} selections, {len(derived)} derived "
          f"plots, {len(variables)} catalog variables")

    regions = raw.get("regions") or {}
    if "regions" in sections and regions:
        print("\nRegions  (histograms are <region>/<cut>/<variable>)")
        for rname, region in regions.items():
            if not region or not any(keep(f"{rname}/{a}") for a in
                                     [""] + list((region.get("objects") or {}))):
                continue
            evt_sel = region.get("evt_sel") or []
            print(f"  {rname}:  evt_sel {evt_sel if evt_sel else '(no cut)'}")
            if region.get("evt_vars"):
                print(f"    evt_vars         {len(region['evt_vars'])} variable(s)")
            for alias, cut in (region.get("objects") or {}).items():
                if not cut:
                    continue
                print(f"    {alias:<16} cut {cut.get('cut') or '(none)'}  "
                      f"{len(cut.get('vars') or [])} variable(s)")

    if "histograms" in sections:
        rows = []
        for name, cfg in hists.items():
            if not keep(name):
                continue
            if is_2d(cfg):
                expr = f"{cfg['expression_x']} : {cfg['expression_y']}"
                binning = f"{_binning(cfg, '_x')} x {_binning(cfg, '_y')}"
            else:
                expr, binning = cfg.get("expression", ""), _binning(cfg)
            region = name.split("/")[0] if "/" in name else None
            files = h_origins.get(f"regions/{region}" if region else name.split("/")[0], [])
            rows.append((name, "2D" if is_2d(cfg) else "1D", expr, binning,
                         ",".join(cfg.get("selections", [])) or "-",
                         " < ".join(_rel(f, base) for f in files) or "?"))
        print(f"\nHistograms ({len(rows)})")
        _table([("name", "dim", "expression", "binning", "selections", "defined in")]
               + rows, (44, 3, 44, 22, 34, 60))

    if "selections" in sections:
        rows = [(name, cfg.get("expression", ""),
                 " < ".join(_rel(f, base) for f in s_origins.get(name, [])) or "?")
                for name, cfg in sels.items() if keep(name)]
        print(f"\nSelections ({len(rows)})")
        _table([("name", "expression", "defined in")] + rows, (36, 80, 60))

    if "variables" in sections:
        rows = [(name, cfg.get("expression", ""), _binning(cfg), cfg.get("label", ""))
                for name, cfg in variables.items() if keep(name)]
        print(f"\nCatalog variables ({len(rows)})")
        _table([("name", "expression", "binning", "label")] + rows, (36, 50, 22, 40))

    if "derived" in sections:
        from .plot import DERIVED_TYPES
        rows = []
        for name, cfg in derived.items():
            if not keep(name):
                continue
            keys = DERIVED_TYPES.get(cfg.get("type", ""), (None, ()))[1]
            rows.append((name, cfg.get("type", "?"),
                         ", ".join(str(cfg.get(k)) for k in keys)))
        print(f"\nDerived plots ({len(rows)})")
        _table([("name", "type", "from")] + rows, (50, 12, 90))
    return 0


# ── check ─────────────────────────────────────────────────────────


def check(config_dirs: list[str], static: bool = False, samples=None) -> int:
    """Static checks, then (unless *static*) every expression on one file per sample."""
    from .processor import (check_expressions, config_warnings, load_config_set,
                            resolve_sample_files, select_samples, validate_config_set)

    rc = 0
    for cdir in config_dirs:
        errors = validate_config_set(cdir)
        warnings = config_warnings(cdir)
        checked = []
        if not static and not errors:
            cs = load_config_set(cdir)
            try:
                chosen = select_samples(cs["samples"], samples)
            except SystemExit as e:
                errors.append(str(e))
                chosen = {}
            remote = any("://" in str(s) for c in chosen.values()
                         for s in (c.get("files") or []))
            if remote:
                from .slurm import ensure_proxy
                ensure_proxy(need_seconds=60)
            for name, cfg in chosen.items():
                try:
                    files = resolve_sample_files(cfg)[:1]
                except Exception as e:  # noqa: BLE001 - reported per sample
                    errors.append(f"[{name}] cannot list its input files: "
                                  f"{type(e).__name__}: {e}")
                    continue
                if not files:
                    errors.append(f"[{name}] no input files resolved")
                    continue
                errors += [f"[{name}] {p}" for p in check_expressions(
                    files, cfg.get("tree", "Events"), cs["histograms"],
                    cs["selections"], cs["corrections"], cs["derive_fn"],
                    cs["reweighters"], cs["settings"], sample=name)]
                checked.append(name)
        status = "FAILED" if errors else "OK"
        detail = (f"; expressions evaluated on one file of {len(checked)} sample(s)"
                  if checked else ("; static checks only" if static else ""))
        print(f"{cdir}: {status}{detail}")
        for e in errors:
            print(f"  ERROR   {e}")
        for w in warnings:
            print(f"  WARNING {w}")
        rc |= bool(errors)
    return rc


# ── fields ────────────────────────────────────────────────────────


def _describe(arr, ranges: bool) -> str:
    import awkward as ak
    import numpy as np

    text = str(arr.type.content) if hasattr(arr.type, "content") else str(arr.type)
    if not ranges:
        return text
    try:
        flat = np.asarray(ak.flatten(arr, axis=None))
    except Exception:  # noqa: BLE001 - records, strings: no range
        return text
    if flat.dtype == bool:
        return f"{text}   true {flat.mean():.1%} of {flat.size}" if flat.size else text
    if not np.issubdtype(flat.dtype, np.number):
        return text
    flat = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.floating) else flat
    if flat.size == 0:
        return f"{text}   (no values)"
    q = np.percentile(flat, [0, 1, 50, 99, 100])
    return (f"{text}   min {q[0]:.4g}  1% {q[1]:.4g}  median {q[2]:.4g}  "
            f"99% {q[3]:.4g}  max {q[4]:.4g}  ({flat.size} values)")


def fields(config_dir: str, sample: str, n: int = 1000, grep: str | None = None,
           ranges: bool = False) -> int:
    """The collections and fields an expression can use, after derive()."""
    from .processor import (load_config_set, open_events, prepare_events,
                            resolve_sample_files, select_samples)

    cs = load_config_set(config_dir)
    cfg = select_samples(cs["samples"], [sample])[sample]
    files = resolve_sample_files(cfg)
    if not files:
        raise SystemExit(f"ERROR: sample '{sample}' resolves no input file")
    if "://" in files[0]:
        from .slurm import ensure_proxy
        ensure_proxy(need_seconds=60)
    print(f"{sample}: first {n} events of {files[0]}")
    events = open_events(files[0], cfg.get("tree", "Events"), n)
    events = prepare_events(events, cs["derive_fn"], cs["settings"],
                            cs["selections"], sample)
    pattern = re.compile(grep) if grep else None
    selections = set(cs["selections"])
    for name in sorted(events.fields):
        sub = events[name]
        sub_fields = list(getattr(sub, "fields", []) or [])
        if sub_fields:
            shown = [f for f in sorted(sub_fields)
                     if pattern is None or pattern.search(f"{name}.{f}")]
            if not shown:
                continue
            print(f"\nevents.{name}  ({'per object' if sub.ndim > 1 else 'per event'})")
            for f in shown:
                print(f"  .{f:<28} {_describe(sub[f], ranges)}")
        elif pattern is None or pattern.search(name):
            tag = "  [selection]" if name in selections else ""
            print(f"events.{name:<30} {_describe(sub, ranges)}{tag}")
    return 0


# ── entry point ───────────────────────────────────────────────────


def main(argv, make_parser, check_config_dir) -> int:
    parser = make_parser(
        "suep-config", "Read and check config sets without running them.",
        epilog="Examples:\n"
               "  suep-config show configs/configs_mds_signal --grep cluster_size\n"
               "  suep-config check configs/configs_mds          # on a compute node:\n"
               "  srun --qos=rapid --time=00:30:00 suep-config check configs/*/\n"
               "  suep-config fields configs/configs_mds -s suep_mDark2_temp1 "
               "--grep cscCluster --ranges")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show", formatter_class=parser.formatter_class,
                       help="what a set fills, and where each definition comes from")
    p.add_argument("config_dir")
    p.add_argument("--grep", default=None,
                   help="only names matching this regular expression")
    p.add_argument("--section", action="append", default=None,
                   choices=["regions", "histograms", "selections", "variables", "derived"],
                   help="show only these sections (repeatable; default: all but variables)")

    p = sub.add_parser("check", formatter_class=parser.formatter_class,
                       help="check sets: references, columns.yaml, and every "
                            "expression on one file per sample")
    p.add_argument("config_dir", nargs="+")
    p.add_argument("--static", action="store_true",
                   help="skip opening input files (no grid proxy needed)")
    p.add_argument("-s", "--samples", nargs="*", default=None,
                   help="check only these samples")

    p = sub.add_parser("fields", formatter_class=parser.formatter_class,
                       help="the collections and fields of a sample after derive()")
    p.add_argument("config_dir")
    p.add_argument("-s", "--sample", required=True)
    p.add_argument("-n", "--events", type=int, default=1000,
                   help="events read from the first file")
    p.add_argument("--grep", default=None,
                   help="only fields matching this regular expression "
                        "(matched against <collection>.<field>)")
    p.add_argument("--ranges", action="store_true",
                   help="also print each field's value range (reads the branches)")

    args = parser.parse_args(argv)
    if args.cmd == "show":
        check_config_dir(args.config_dir, runnable=False)
        return show(args.config_dir, args.grep, args.section)
    if args.cmd == "check":
        from .config import read_set_file
        dirs = []
        for cdir in args.config_dir:
            check_config_dir(cdir, runnable=False)
            if read_set_file(cdir).get("abstract"):
                print(f"{cdir}: base set (abstract), checked through the sets "
                      "that extend it")
                continue
            dirs.append(cdir)
        return check(dirs, args.static, args.samples)
    check_config_dir(args.config_dir)
    return fields(args.config_dir, args.sample, args.events, args.grep, args.ranges)
