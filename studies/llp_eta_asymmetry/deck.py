"""Build the discussion deck for the SUEP dark-meson eta asymmetry."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

NAVY = RGBColor(0x12, 0x20, 0x3F)
INK = RGBColor(0x1B, 0x2A, 0x41)
TEAL = RGBColor(0x1C, 0x72, 0x93)
ALERT = RGBColor(0xE4, 0x57, 0x2E)
MUTED = RGBColor(0x6B, 0x7A, 0x8F)
ICE = RGBColor(0xE8, 0xEE, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xE8, 0xEE, 0xF5)

HEAD, BODY, MONO = "Cambria", "Calibri", "Courier New"
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    if dark:
        bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()
        bg.shadow.inherit = False
    return s


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: list of (string, size_pt, bold, color, font, space_after_pt)."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, (t, size, bold, color, font, after) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = color
    return box


def title(s, txt, dark=False, step=None):
    """Slide title, optionally preceded by a numbered step badge."""
    x = 0.75
    if step is not None:
        d = 0.52
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(0.62),
                               Inches(d), Inches(d))
        c.fill.solid()
        c.fill.fore_color.rgb = TEAL
        c.line.fill.background()
        c.shadow.inherit = False
        tf = c.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = str(step)
        r.font.size, r.font.bold, r.font.name = Pt(20), True, HEAD
        r.font.color.rgb = WHITE
        x += d + 0.24
    text(s, x, 0.6, W - x - 0.75, 0.72,
         [(txt, 30, True, WHITE if dark else INK, HEAD, 0)])


def card(s, x, y, w, h, body_runs, tint=PALE):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = tint
    r.line.fill.background()
    r.shadow.inherit = False
    r.adjustments[0] = 0.06
    r.text_frame.word_wrap = True
    text(s, x + 0.28, y + 0.22, w - 0.56, h - 0.44, body_runs)
    return r


def bullets(s, x, y, w, items, size=14, gap=0.62, color=INK):
    """Small circular markers + text, the deck's repeated list motif."""
    for i, item in enumerate(items):
        yy = y + i * gap
        d = 0.11
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(yy + 0.07),
                               Inches(d), Inches(d))
        c.fill.solid()
        c.fill.fore_color.rgb = TEAL
        c.line.fill.background()
        c.shadow.inherit = False
        text(s, x + 0.3, yy, w - 0.3, gap, [(item, size, False, color, BODY, 0)])


def stat(s, x, y, w, value, label, color=ALERT):
    text(s, x, y, w, 0.62, [(value, 34, True, color, HEAD, 0)])
    text(s, x, y + 0.58, w, 0.5, [(label, 11.5, False, MUTED, BODY, 0)])


def pic(s, name, x, y, w):
    return s.shapes.add_picture(name, Inches(x), Inches(y), width=Inches(w))


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt


# ── 1. title ──────────────────────────────────────────────────────────
s = slide(dark=True)
text(s, 0.9, 2.05, 11.6, 0.5,
     [("GENERATOR-LEVEL FINDING", 13, True, RGBColor(0x7F, 0xB2, 0xCB), BODY, 0)])
text(s, 0.9, 2.62, 11.4, 1.9,
     [("A forward–backward asymmetry\nin the SUEP dark mesons", 40, True, WHITE, HEAD, 0)])
text(s, 0.9, 4.65, 10.6, 0.9,
     [("The dark mesons in the private SUEPs_Gen2 2024 samples are emitted "
       "preferentially towards lab −z. Traced to a one-character typo in the "
       "CMSSW SUEP shower, fixed upstream in CMSSW_15_0_15.",
       15, False, ICE, BODY, 0)])
text(s, 0.9, 6.35, 11.5, 0.4,
     [("Ang Li  ·  29 July 2026  ·  ggH, mMed = 125 GeV, mDark = 2 GeV, "
       "cτ = 5 m  ·  60k events (T = 1, T = 2)", 11.5, False,
       RGBColor(0x8E, 0xA3, 0xBC), BODY, 0)])
note(s, "Question raised: why is the LLP eta distribution asymmetric? "
        "Answer: it is real, it is in the gen record, and it looks like a "
        "generator bug.")

# ── 2. the observation ────────────────────────────────────────────────
s = slide()
title(s, "The observation")
bullets(s, 0.78, 1.6, 6.0, [
    "Gen dark mesons (SUEPGenPart, pdgId 999999) sit preferentially at "
    "negative η, in every file and at both temperatures.",
    "Mirroring the histogram makes it plain: h(η)/h(−η) slopes "
    "from about 1.25 down to 0.75 across the range.",
    "pp collisions have no preferred z direction, so this cannot be physical.",
], gap=1.02)
stat(s, 0.78, 5.0, 2.0, "−0.158", "⟨η⟩, T = 1")
stat(s, 2.85, 5.0, 2.0, "−0.217", "⟨η⟩, T = 2")
stat(s, 4.92, 5.0, 2.2, "≥ 36σ", "even counting each event\nas one measurement", TEAL)
pic(s, "fig1_lab_eta.png", 7.35, 1.45, 5.25)
note(s, "Blue/red are the two temperatures, black dashed is the same "
        "histogram mirrored about zero. The ratio panel is the asymmetry.")

# ── 3. notation ───────────────────────────────────────────────────────
s = slide()
title(s, "Notation: the starred quantities")
defs = [
    ("S", "the summed 4-momentum of all the dark mesons in an event — equal "
          "to the Higgs to 1.8 GeV rms."),
    ("star", "measured in the rest frame of S. The boost is a pure boost, so "
             "the starred axes stay parallel to the lab ones."),
    ("p*", "the momentum magnitude of one meson in that frame: how hard the "
           "meson is. ⟨p*⟩ = 3.4 GeV."),
    ("cosθ*_z = p*_z / p*", "where that meson points relative to the beam. "
                            "+1 along +z, 0 transverse, −1 along −z."),
    ("cosθ*_x , cosθ*_y", "the same construction against the lab x and y "
                          "axes — the control directions."),
    ("corr(p*, cosθ*_z)", "Pearson coefficient over all 1.24M mesons: 0 means "
                          "the direction tells you nothing about the "
                          "momentum, ±1 means it fixes it."),
]
for j, (sym, meaning) in enumerate(defs):
    y = 1.45 + j * 0.88
    text(s, 0.78, y, 6.0, 0.3, [(sym, 14, True, TEAL, HEAD, 0)])
    text(s, 0.78, y + 0.32, 6.0, 0.5, [(meaning, 12.5, False, INK, BODY, 0)])
pic(s, "fig0_notation.png", 7.1, 1.55, 5.55)
card(s, 7.1, 4.2, 5.55, 1.6,
     [("If the mesons were emitted isotropically, all three cosines would be "
       "flat on [−1, +1] and uncorrelated with p*. That is the null hypothesis "
       "every plot in this deck is tested against.", 13, False, INK, BODY, 0)])
note(s, "Everything starred lives in the rest frame of the meson system. "
        "The axes are still the lab axes — that is what makes a z-preference "
        "meaningful at all.")

# ── 4. the test chain ─────────────────────────────────────────────────
s = slide()
title(s, "Six things that would have to be true")
text(s, 0.78, 1.42, 11.8, 0.4,
     [("Each check below is a way the asymmetry could have been mundane. "
       "None of them survived.", 14.5, False, MUTED, BODY, 0)])
checks = [
    ("1", "Is the parent Higgs asymmetric?", "No — ⟨ηₕ⟩ = +0.004 ± 0.014"),
    ("2", "Is 4-momentum conserved?", "Yes — exactly, m = 124.9995 GeV"),
    ("3", "Is the decay isotropic at rest?", "Only in x and y, not in z"),
    ("4", "Does the axis follow the parent?", "No — it is the fixed lab z"),
    ("5", "Is |p*| tied to the direction?", "Yes — strongly, corr = 0.72"),
    ("6", "Could it be a frame error?", "No — nothing restores isotropy"),
]
for i, (num, q, a) in enumerate(checks):
    col, row = i % 3, i // 3
    x, y = 0.78 + col * 4.02, 2.15 + row * 2.35
    card(s, x, y, 3.72, 2.0, [])
    d = 0.46
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.28), Inches(y + 0.26),
                           Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = TEAL
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = num
    r.font.size, r.font.bold, r.font.name = Pt(16), True, HEAD
    r.font.color.rgb = WHITE
    text(s, x + 0.28, y + 0.86, 3.16, 0.52, [(q, 14, True, INK, BODY, 0)])
    text(s, x + 0.28, y + 1.44, 3.16, 0.44, [(a, 12.5, False, ALERT, BODY, 0)])
note(s, "Roadmap for the next six slides.")

# ── 4. check 1 ────────────────────────────────────────────────────────
s = slide()
title(s, "The parent Higgs is clean", step=1)
bullets(s, 0.78, 1.75, 5.9, [
    "The last-copy Higgs (status 62) is symmetric to the precision of the sample.",
    "⟨ηₕ⟩ = +0.004 ± 0.014 and frac(ηₕ < 0) = 0.4983 — both consistent with zero.",
    "So production is fine. Whatever happens, happens in the decay.",
], gap=1.02)
card(s, 0.78, 5.05, 5.9, 1.0,
     [("Rules out: a biased ggH production or a beam-configuration problem.",
       13.5, False, INK, BODY, 0)])
pic(s, "fig2_higgs_eta.png", 7.35, 1.45, 5.05)
note(s, "Same mirror test as the previous slide, applied to the Higgs.")

# ── 5. check 2 ────────────────────────────────────────────────────────
s = slide()
title(s, "4-momentum closes exactly", step=2)
bullets(s, 0.78, 1.45, 11.8, [
    "Σᵢ pᵢ(mesons) − p(H) is consistent with zero in every component "
    "(⟨Δp_z⟩ = −0.019 GeV).",
    "The invariant mass of the dark-meson system is 124.9995 GeV.",
    "Rules out a broken boost, a truncated collection, or missing and "
    "double-counted mesons — the set we are looking at is complete.",
], gap=0.58, size=14.5)
pic(s, "fig3_closure.png", 1.47, 3.28, 10.4)
note(s, "Left panel is log scale; all four residual components peak at zero.")

# ── 6. check 3 ────────────────────────────────────────────────────────
s = slide()
title(s, "In the rest frame: isotropic in x and y, not in z", step=3)
bullets(s, 0.78, 1.75, 5.9, [
    "Boosting to the frame where the meson system is at rest, the direction "
    "cosines along x and y are flat and centred on zero.",
    "Along z they are not: ⟨cosθ*_z⟩ = −0.1822, a 420σ departure.",
    "The +z pole is essentially empty — 14691 mesons in the first bin "
    "against 15 in the last.",
], gap=1.12)
stat(s, 0.78, 5.35, 2.3, "−0.1822", "⟨cosθ*_z⟩")
stat(s, 3.2, 5.35, 2.3, "−0.0000", "⟨cosθ*_x⟩, ⟨cosθ*_y⟩", TEAL)
pic(s, "fig4_restframe_cos.png", 7.15, 1.7, 5.45)
note(s, "This is the core plot: the anisotropy singles out one axis.")

# ── 7. check 4 ────────────────────────────────────────────────────────
s = slide()
title(s, "The axis is the fixed lab z, not the parent", step=4)
bullets(s, 0.78, 1.75, 5.9, [
    "Split the mesons by the sign of their own parent's rapidity.",
    "ηₕ < 0 gives ⟨cosθ*_z⟩ = −0.1823; ηₕ > 0 gives −0.1821. Identical.",
    "A physical decay asymmetry would have to flip with the parent. This one "
    "does not — it points at lab −z regardless.",
], gap=1.02)
card(s, 0.78, 5.25, 5.9, 1.15,
     [("This is the step that makes a physics explanation impossible: no decay "
       "can know about the laboratory z axis.", 13.5, False, INK, BODY, 0)])
pic(s, "fig5_cos_by_parent.png", 7.35, 1.7, 5.05)
note(s, "The two curves lie on top of each other.")

# ── 8. check 5 ────────────────────────────────────────────────────────
s = slide()
title(s, "Soft mesons go −z, hard mesons go +z", step=5)
bullets(s, 0.78, 1.45, 11.8, [
    "|p*| and cosθ*_z are almost one-to-one: the profile runs from −0.68 "
    "at low p* to +0.59 at high p*, with a sharp edge near cosθ*_z ≈ 0.6.",
    "The momentum-weighted ⟨cosθ*_z⟩ is 0.000000 — the momenta still "
    "balance. Over all mesons, corr(p*, cosθ*_z) = +0.72.",
    "The imbalance is in the number of mesons, not in the momentum. That is "
    "why η is skewed while the total still adds up to the Higgs.",
], gap=0.58, size=14.5)
pic(s, "fig6_cos_vs_pstar.png", 1.47, 3.28, 10.4)
note(s, "Left: 2D density. Right: profile of the mean.")

# ── 9. check 6 ────────────────────────────────────────────────────────
s = slide()
title(s, "No boost needed to see it", step=6)
bullets(s, 0.78, 1.75, 5.9, [
    "Take only events where the meson system is already at rest along z, "
    "|p_z| < 10 GeV. The lab frame is then the rest frame.",
    "With no boost applied anywhere, the raw lab η is still 57.5% negative, "
    "⟨η⟩ = −0.278.",
    "So the effect is not something our boost or our column code introduces.",
], gap=1.02)
card(s, 0.78, 5.25, 5.9, 1.15,
     [("Rules out: any suspicion that the asymmetry is manufactured by the "
       "rest-frame transformation used in the previous slides.",
       13.5, False, INK, BODY, 0)])
pic(s, "fig7_no_boost.png", 7.35, 1.45, 5.15)
note(s, "The cleanest statement: raw numbers straight out of the file.")

# ── 10. not a frame error ─────────────────────────────────────────────
s = slide()
title(s, "It is not a bookkeeping error")
bullets(s, 0.78, 1.45, 11.8, [
    "Scan a longitudinal boost β_z, and separately a longitudinal momentum "
    "offset Δp_z, applied to the rest-frame momenta.",
    "⟨cosθ*_z⟩ can be driven to zero by either, but corr(p*, cosθ*_z) — how "
    "well a meson's direction predicts its momentum — never falls below "
    "about 0.47, and the distribution never becomes flat.",
    "No choice of frame or shift makes the sample isotropic — the sampling of "
    "the meson directions is itself z-dependent.",
], gap=0.58, size=14.5)
pic(s, "fig9_not_a_frame_error.png", 1.47, 3.28, 10.4)
note(s, "We tried to explain it away as a frame convention. It does not work.")

# ── 11. impact ────────────────────────────────────────────────────────
s = slide()
title(s, "What it costs us")
bullets(s, 0.78, 1.45, 11.8, [
    "The reconstructed muon-detector clusters inherit the asymmetry directly.",
], gap=0.5, size=14.5)
pic(s, "fig8_reco_clusters.png", 0.78, 2.0, 8.6)
card(s, 9.6, 2.0, 2.95, 3.06,
     [("Any per-endcap efficiency or acceptance from these samples is biased "
       "at the few-percent level.\n\nFolding into |η| is not a repair — see "
       "the next two slides.", 13, False, INK, BODY, 0)], tint=ICE)
for i, (v, l) in enumerate([("+0.015", "CSC, T = 1"), ("+0.072", "CSC, T = 2"),
                            ("−0.014", "DT, T = 1"), ("+0.019", "DT, T = 2")]):
    stat(s, 0.78 + i * 2.22, 5.45, 2.1, v, l)
note(s, "A = (N₋ − N₊)/(N₋ + N₊) quoted under each sample.")

# ── 12. root cause ────────────────────────────────────────────────────
s = slide()
title(s, "Root cause: one character in SuepShower.cc")
bullets(s, 0.78, 1.5, 6.5, [
    "The mediator decay is replaced by the SuepDecay Pythia hook, which calls "
    "SuepShower::generateFourVector() once per dark meson: draw a thermal |p|, "
    "then draw isotropic angles.",
    "The z component of that 4-vector is built with sin(θ) where it should be "
    "cos(θ). Every meson is therefore generated with p_z ≥ 0.",
    "generateShower() then subtracts the mean 3-momentum from every daughter, "
    "which pushes the bulk of them to negative p_z — and leaves the hard ones "
    "at +z. That is the whole effect.",
], gap=1.28)
card(s, 7.75, 1.5, 4.8, 2.35,
     [("Already fixed upstream.\n\nThe one-line patch landed in CMSSW_15_0_15 "
       "(1 Oct 2025) and in 15_1_0. Everything from 14_0_0 through 15_0_14 "
       "carries the bug — our samples were made with 15_0_2.",
       13, False, INK, BODY, 0)], tint=ICE)
card(s, 0.78, 5.35, 11.75, 1.55, [], tint=PALE)
text(s, 1.06, 5.55, 11.2, 0.3,
     [("GeneratorInterface/Pythia8Interface/src/SuepShower.cc : 158", 11,
       True, MUTED, BODY, 0)])
text(s, 1.06, 5.92, 11.2, 0.34,
     [("−   Vec4(p*cos(phi)*sin(theta), p*sin(phi)*sin(theta), p*sin(theta), en)",
       12, False, ALERT, MONO, 0)])
text(s, 1.06, 6.32, 11.2, 0.34,
     [("+   Vec4(p*cos(phi)*sin(theta), p*sin(phi)*sin(theta), p*cos(theta), en)",
       12, False, TEAL, MONO, 0)])
note(s, "A copy-paste slip: the transverse components are right, the "
        "longitudinal one repeats sin(theta). Note it also means the vector "
        "magnitude is not the sampled |p| — the momentum spectrum is distorted "
        "too, not only the angles.")

# ── 13. toy validation ────────────────────────────────────────────────
s = slide()
title(s, "The typo reproduces the samples exactly")
bullets(s, 0.78, 1.42, 11.8, [
    "A toy re-implementation of SuepShower with the typo matches the samples "
    "bin for bin: ⟨cosθ*_z⟩ = −0.184 against −0.182 measured, multiplicity "
    "31.0 against 31.0, ⟨p*⟩ = 3.42 against 3.41 GeV. With cos(θ) it gives "
    "−0.0003 and a flat distribution.",
    "Folding into |η| removes the forward/backward asymmetry but not the "
    "distortion: the folded angular distribution is still wrong by up to 50%. "
    "These samples cannot be repaired after the fact — they need regenerating.",
], gap=0.72, size=14.5)
pic(s, "fig10_rootcause.png", 1.47, 3.15, 10.4)
note(s, "Left: data points on top of the toy. Right: why |eta|-folding is a "
        "cosmetic fix only.")

# ── 14. summary ───────────────────────────────────────────────────────
s = slide(dark=True)
title(s, "Where that leaves us", dark=True)
text(s, 0.78, 1.55, 5.6, 0.4,
     [("ESTABLISHED", 12.5, True, RGBColor(0x7F, 0xB2, 0xCB), BODY, 0)])
bullets(s, 0.78, 2.05, 5.5, [
    "The asymmetry is real, gen-level, and about 5–10% in the meson yield.",
    "Cause: SuepShower.cc builds p_z with sin(θ) instead of cos(θ).",
    "A toy with that one line reproduces the samples bin for bin.",
    "Fixed upstream in CMSSW_15_0_15; our samples used 15_0_2.",
], gap=0.9, color=ICE)
text(s, 6.95, 1.55, 5.6, 0.4,
     [("OPEN FOR DISCUSSION", 12.5, True, RGBColor(0xF0, 0x9C, 0x7E), BODY, 0)])
bullets(s, 6.95, 2.05, 5.5, [
    "When do we regenerate? Every point, or only what the analysis needs?",
    "Move the GEN step to CMSSW_15_0_15+, or patch 15_0_2 in place?",
    "Which other SUEP samples — central productions, other analyses — were "
    "made with an affected release?",
    "Anything already shown from these samples needs revisiting.",
], gap=0.9, color=ICE)
text(s, 0.78, 6.55, 11.8, 0.4,
     [("Diagnostics reproducible from the MDSNANO files: 3 files, 60k events, "
       "gen-level only.  ·  Root cause confirmed against a toy of the "
       "generator.", 11.5, False, RGBColor(0x8E, 0xA3, 0xBC), BODY, 0)])
note(s, "Ask the generator experts to look at the momentum-sampling step.")

prs.save("SUEP_LLP_eta_asymmetry.pptx")
print("wrote SUEP_LLP_eta_asymmetry.pptx", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
