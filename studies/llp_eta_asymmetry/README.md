# LLP eta asymmetry in the SUEPs_Gen2 2024 samples

Gen-level diagnosis of the forward/backward asymmetry seen in `llp_eta`
(and inherited by the CSC/DT cluster eta distributions).

**Finding:** the dark mesons (`SUEPGenPart`, pdgId 999999) are emitted
preferentially towards lab −z. The parent Higgs is symmetric and 4-momentum is
conserved exactly, so this is not production and not a broken boost.

**Root cause:** `GeneratorInterface/Pythia8Interface/src/SuepShower.cc:158`
composes the daughter 4-vector as

```
Vec4(p*cos(phi)*sin(theta), p*sin(phi)*sin(theta), p*sin(theta), en)
                                                     ^^^ must be cos(theta)
```

so every dark meson is generated with `p_z >= 0`. `generateShower()` then
subtracts the mean 3-momentum from each daughter, which pushes most of them to
negative `p_z` and leaves the hard ones at `+z`. Fixed upstream in
**CMSSW_15_0_15** (1 Oct 2025) and 15_1_0; every release from 14_0_0 through
15_0_14 carries it. These samples were produced with 15_0_2.

**Consequence:** the samples must be regenerated. Folding into `|eta|` removes
the forward/backward asymmetry but not the underlying angular distortion, which
is still up to 50% away from isotropic after folding.

## Reproducing

Needs the `mds` conda env. The MDSNANO files are read from the local EOS mount,
so no grid proxy is required.

```bash
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python extract.py
```

`extract.py` reads 3 files (60k events, T=1 and T=2) and caches the flat arrays
to `gen_asymmetry.npz`. Then:

```bash
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python plots.py
```

`plots.py` writes `fig1`–`fig9` and prints every number quoted on the slides.
Figure 8 is read from `../../output_mds_llp/*.pkl`, not from the npz. The
notation schematic is standalone and needs no input:

```bash
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python fig0.py
```

`toy.py` re-implements `SuepShower` in ~60 lines and runs it with and without
the typo; `fig10.py` overlays that toy on the real samples. Both are standalone
apart from `gen_asymmetry.npz`:

```bash
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python toy.py
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python fig10.py
```

```bash
/groups/hephy/cms/ang.li/anaconda3/envs/mds/bin/python deck.py
```

`deck.py` assembles `SUEP_LLP_eta_asymmetry.pptx` from the figures. It needs
`python-pptx`, and it expects the figures in the working directory.

## The figures

| | |
|---|---|
| `fig0_notation` | schematic of the boost and of `p*`, `cosθ*_z` |
| `fig1_lab_eta` | the observation: gen meson eta, overlaid with its own mirror |
| `fig2_higgs_eta` | the parent Higgs is symmetric |
| `fig3_closure` | Σp(mesons) − p(H) and the system invariant mass |
| `fig4_restframe_cos` | rest-frame direction cosines: flat in x and y, not in z |
| `fig5_cos_by_parent` | the anisotropy does not follow the parent's rapidity |
| `fig6_cos_vs_pstar` | soft mesons go −z, hard mesons go +z |
| `fig7_no_boost` | same asymmetry in events where the system is already at rest |
| `fig8_reco_clusters` | what the CSC and DT clusters inherit |
| `fig9_not_a_frame_error` | no boost or momentum offset restores isotropy |
| `fig10_rootcause` | the toy with the typo, on top of the samples; and why folding does not repair them |
