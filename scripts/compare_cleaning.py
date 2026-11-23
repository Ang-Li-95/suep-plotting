"""Compare the overlay's MC cleaning with the DY noise study, number for number.

    python scripts/compare_cleaning.py <run_dir>

*run_dir* is a ``suep-run`` / ``suep-submit`` output of
``configs/configs_noise_cleaning``: the study's own premix_truth events,
cleaned by ``suep_plot.overlay``.  For every sample and system this prints the
efficiency (kept hard-process rechits / hard-process rechits), the purity
(kept hard / kept) and the kept non-hard rechits per event, next to what the
study's ``plot_matching_performance.py`` reported, rounded as the study rounds
them.  The exit status is 1 if any number differs.
"""

import argparse
import sys

from suep_plot.results import merge_results, resolve_inputs

# plot_matching_performance.py (GenJet pT > 10) on 2026-10-18, as logged in
# $SCRATCHDIR/DYNoiseStudy/matching_perf/slurm3.log and drawn in
# suep-plotting/output_dy_noise/plots/matching_performance/summary.png.
REFERENCE = {
    "dy": {"CSC": (0.9673, 0.9281, 1.703), "DT": (0.9944, 0.977, 0.551),
           "RPC": (0.9916, 0.9854, 0.061)},
    "tt": {"CSC": (0.8979, 0.86, 2.014), "DT": (0.9441, 0.9037, 1.758),
           "RPC": (0.9394, 0.9429, 0.164)},
    "qcd": {"CSC": (0.8479, 0.8521, 1.797), "DT": (0.7966, 0.7859, 1.752),
            "RPC": (0.8048, 0.8906, 0.134)},
}
REFERENCE_EVENTS = {"dy": 2000, "tt": 2000, "qcd": 2075}
SAMPLES = {"noise_truth_dy": "dy", "noise_truth_tt": "tt", "noise_truth_qcd": "qcd"}
SYSTEMS = {"CSC": "csc", "DT": "dt", "RPC": "rpc"}
# The unit weights are genWeight * (1 / genWeight) with a float32 genWeight,
# so each is 1 only to ~1e-7.  A summed count further than this fraction from
# an integer means the histograms are not unit-weighted at all.
RELATIVE_INTEGER_TOLERANCE = 1e-6


def rechit_count(histograms, sample, system, cut):
    """Number of MC rechits of *system* passing *cut*, from its eta histogram."""
    h = histograms[f"{system}/{cut}/{system}_mc_rechit_eta"]
    total = h[{"dataset": sample}].sum(flow=True).value
    count = round(total)
    if abs(total - count) > RELATIVE_INTEGER_TOLERANCE * max(count, 1):
        raise SystemExit(f"ERROR: {sample} {system}/{cut}: weighted count {total!r} "
                         "is not an integer; are the histograms unit-weighted?")
    return count


def measure(histograms, sample, n_events):
    """{SYSTEM: (efficiency, purity, non-hard kept per event)}, rounded as the study."""
    out = {}
    for system_label, system in SYSTEMS.items():
        hard = rechit_count(histograms, sample, system, "hard")
        kept = rechit_count(histograms, sample, system, "kept")
        hard_kept = rechit_count(histograms, sample, system, "hard_kept")
        nonhard_kept = rechit_count(histograms, sample, system, "nonhard_kept")
        out[system_label] = (round(hard_kept / hard, 4), round(hard_kept / kept, 4),
                             round(nonhard_kept / n_events, 3))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()

    result = merge_results(resolve_inputs(args.run_dir))
    histograms, nevents = result["histograms"], result["nevents"]

    mismatches = 0
    print(f"{'sample':6s} {'system':6s} {'quantity':22s} {'suep-plotting':>14s} "
          f"{'study':>8s}")
    for sample, short in SAMPLES.items():
        n_events = nevents.get(sample, 0)
        same = n_events == REFERENCE_EVENTS[short]
        mismatches += not same
        print(f"{short:6s} {'':6s} {'events':22s} {n_events:14d} "
              f"{REFERENCE_EVENTS[short]:8d}{'' if same else '   <-- differs'}")
        measured = measure(histograms, sample, n_events)
        for system, values in measured.items():
            for quantity, ours, theirs in zip(
                    ("efficiency", "purity", "non-hard kept / event"),
                    values, REFERENCE[short][system]):
                same = ours == theirs
                mismatches += not same
                print(f"{short:6s} {system:6s} {quantity:22s} {ours:14.4f} "
                      f"{theirs:8.4f}{'' if same else '   <-- differs'}")

    print(f"\n{mismatches} difference(s)")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
