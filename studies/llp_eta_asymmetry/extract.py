"""Dump the gen-level quantities needed for the LLP eta-asymmetry diagnostics.

One pass over a few MDSNANO files, flat arrays cached to an npz so the plotting
script can be re-run without touching xrootd again.
"""
import numpy as np
import awkward as ak
import uproot
import vector

vector.register_awkward()

EOS = ("/eos/vbc/experiments/cms/store/user/lian/"
       "SUEPs_Gen2/2024/run3_ggH-channel_mMed-125_mDark-2_portal-higgs_"
       "{temp}_decay-generic_ctau-5000p0_13TeV-pythia8/MDSNANO/nano_{n}.root")
SAMPLES = {"T1": [("temp-1", 0), ("temp-1", 7)], "T2": [("temp-2", 0)]}
LLP_PDGID = 999999

BRANCHES = ["GenPart_pdgId", "GenPart_status", "GenPart_pt", "GenPart_eta",
            "GenPart_phi", "GenPart_mass", "SUEPGenPart_pdgId",
            "SUEPGenPart_pt", "SUEPGenPart_eta", "SUEPGenPart_phi",
            "SUEPGenPart_mass"]


def process(path):
    a = uproot.open(path)["Events"].arrays(BRANCHES)

    # Last-copy Higgs (status 62): the mediator whose decay products the SUEP is.
    hmask = (a["GenPart_pdgId"] == 25) & (a["GenPart_status"] == 62)
    H = ak.firsts(ak.zip({k: a["GenPart_" + k][hmask]
                          for k in ("pt", "eta", "phi", "mass")},
                         with_name="Momentum4D"))

    smask = a["SUEPGenPart_pdgId"] == LLP_PDGID
    L = ak.zip({k: a["SUEPGenPart_" + k][smask]
                for k in ("pt", "eta", "phi", "mass")}, with_name="Momentum4D")

    # SUEP system built from the mesons themselves, so the rest-frame boost
    # never relies on the Higgs being the right parent.
    S = ak.zip({"px": ak.sum(L.px, axis=1), "py": ak.sum(L.py, axis=1),
                "pz": ak.sum(L.pz, axis=1), "E": ak.sum(L.E, axis=1)},
               with_name="Momentum4D")
    Lr = L.boostCM_of(S)

    n = ak.to_numpy(ak.num(L))
    flat = lambda x: ak.to_numpy(ak.flatten(x))
    return dict(
        n_llp=n,
        h_eta=ak.to_numpy(H.eta), h_pt=ak.to_numpy(H.pt),
        h_pz=ak.to_numpy(H.pz),
        s_pz=ak.to_numpy(S.pz), s_mass=ak.to_numpy(S.mass),
        d_px=ak.to_numpy(S.px - H.px), d_py=ak.to_numpy(S.py - H.py),
        d_pz=ak.to_numpy(S.pz - H.pz), d_E=ak.to_numpy(S.E - H.E),
        lab_eta=flat(L.eta), lab_pt=flat(L.pt),
        cos_x=flat(Lr.px / Lr.p), cos_y=flat(Lr.py / Lr.p),
        cos_z=flat(Lr.pz / Lr.p), p_star=flat(Lr.p),
        parent_eta=np.repeat(ak.to_numpy(H.eta), n),
        parent_spz=np.repeat(ak.to_numpy(S.pz), n),
    )


out = {}
for tag, files in SAMPLES.items():
    parts = [process(EOS.format(temp=t, n=n)) for t, n in files]
    for key in parts[0]:
        out[f"{tag}_{key}"] = np.concatenate([p[key] for p in parts])
    print(tag, "events", len(out[f"{tag}_h_eta"]), "LLPs", len(out[f"{tag}_lab_eta"]))

np.savez_compressed("gen_asymmetry.npz", **out)
print("wrote gen_asymmetry.npz")
