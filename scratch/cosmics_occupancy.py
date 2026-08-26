import sys, numpy as np, uproot, awkward as ak, yaml
sys.path.insert(0, "/users/ang.li/public/SUEP/suep-plotting/src")
from suep_plot.processor import _resolve_files
reg = yaml.safe_load(open("/users/ang.li/public/SUEP/suep-plotting/datasets.yaml"))
files = _resolve_files(reg["cosmics_2024C"]["files"])
# the two files whose tasks ran out of memory
for idx in (10, 75, 0):
    t = uproot.open(files[idx])["Events"]
    a = t.arrays(["ncscRechits", "ndtRecHits", "nrpcRecHits"], entry_stop=200000)
    for br in ("ncscRechits", "ndtRecHits", "nrpcRecHits"):
        v = ak.to_numpy(a[br])
        gb = (v.astype(float) ** 2 * 8) / 1e9
        print("  file %3d %-13s max=%7d  p99.9=%7.0f  matrix at max=%6.1f GB  n>20k: %d"
              % (idx, br, v.max(), np.percentile(v, 99.9), gb.max(), (v > 20000).sum()))
    print()
