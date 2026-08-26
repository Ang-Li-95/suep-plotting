"""Merge each output directory and report what landed."""
import sys, glob
sys.path.insert(0, "/users/ang.li/public/SUEP/suep-plotting/src")
from suep_plot.plot import merge_results

B = "/groups/hephy/cms/ang.li/suep_output"
for d in sys.argv[1:]:
    files = sorted(glob.glob(f"{B}/{d}/*.pkl"))
    if not files:
        print(f"{d}: no pickles"); continue
    m = merge_results(files)
    h = m["histograms"]
    nonempty = sum(1 for k in h if h[k].view(flow=True)["value"].sum() > 0)
    print(f"{d}: {len(files)} parts, {len(h)} histograms ({nonempty} non-empty)")
    for s, n in sorted(m["nevents"].items()):
        print("    %-22s %12d events" % (s, n))
