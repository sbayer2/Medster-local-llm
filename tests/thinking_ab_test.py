"""Factorial isolation test for the analyze_document selective-thinking path.

Question: does thinking-ON causally improve the clinical analysis, with the
PROMPT and TEMPERATURE controlled? The shipped comparison (comprehensive vs
complicated) confounds three variables at once — prompt template, thinking flag,
AND temperature — so it can't answer that. This holds the prompt fixed
(_COMPLICATED_ANALYSIS_PROMPT for every cell) and grids only the knobs.

Cells (prompt fixed):
    A  think OFF  temp 0    max_tokens 1024   x1   deterministic baseline
    B  think ON   temp 0    max_tokens 1024   x1   isolates THINKING  (A vs B)
    C  think OFF  temp 0.4  max_tokens 1024   x5   isolates TEMPERATURE
    D  think ON   temp 0.4  max_tokens 1024   x5   the shipped 'complicated'
    E  think ON   temp 0    max_tokens 2048   x1   does raising the cap rescue it?

Why max_tokens matters: _vision_generate defaults max_tokens=1024 and
analyze_document does NOT override it, so the thinking preamble and the clinical
answer share ONE fixed 1024-token budget. Think-on therefore doesn't just leak
cosmetically — it starves the answer. Cell E tests whether more budget fixes it.

Calls _vision_generate directly (read-only import). No src/ changes.
Full outputs dumped to tests/ab_out/ for eyeball grading (the auto-grader below
is a screen only — we learned substring matching undercounts).

    uv run python tests/thinking_ab_test.py
"""
import os
import re
import time
import statistics
import mlx.core as mx

from medster.tools.analysis.primitives import _vision_generate
from medster.tools.analysis.document_analyzer import (
    _COMPLICATED_ANALYSIS_PROMPT,
    SYNTHETIC_DATA_DISCLAIMER,
)

NOTE = (
    "CC: 52M, 8 months fatigue + hand joint pain. "
    "HPI: progressive fatigue; bilateral 2nd-3rd MCP arthralgia; low libido/ED; "
    "slate-bronze skin darkening including non-sun-exposed areas; recent polyuria/polydipsia. "
    "PMH: T2DM diagnosed 3 months ago (A1c 8.9%); 'fatty liver' on ultrasound; mild depression. "
    "MEDS: metformin 1000 BID; sertraline 50; started OTC ferrous sulfate 325 mg BID 6 weeks ago for fatigue. "
    "SOCIAL: 3-4 beers nightly x20 yrs; outdoor construction. "
    "FHX: father died 58 of 'liver problems'; sister with 'arthritis'. "
    "EXAM: diffuse bronze hyperpigmentation incl non-sun-exposed skin; firm hepatomegaly 4 cm below "
    "costal margin, nontender; tender squared 2nd-3rd MCPs; small soft testes; no ascites/edema. "
    "LABS: AST 78, ALT 65, ALP 110, T.bili 1.1; glucose 210, A1c 8.9%; FERRITIN 2150; "
    "TRANSFERRIN SAT 82%; serum iron high, TIBC low; Hgb 14.8 (not anemic); TSH normal; "
    "testosterone low; ECG low-voltage; echo mild concentric LV changes. "
    "ASSESSMENT/PLAN (covering provider): (1) T2DM-optimize metformin; (2) hand OA-NSAIDs PRN; "
    "(3) NAFLD from alcohol/obesity-counsel EtOH reduction; (4) fatigue/low libido-continue ferrous "
    "sulfate, consider starting testosterone; (5) depression-continue sertraline; f/u 3 months."
)

PROMPT = _COMPLICATED_ANALYSIS_PROMPT.format(
    synthetic_disclaimer=SYNTHETIC_DATA_DISCLAIMER, note_text=NOTE, context=""
)

# (label, enable_thinking, temperature, max_tokens, [seeds])
CELLS = [
    ("A_off_t0_1024",  False, 0.0, 1024, [0]),
    ("B_on_t0_1024",   True,  0.0, 1024, [0]),
    ("C_off_t04_1024", False, 0.4, 1024, [1, 2, 3, 4, 5]),
    ("D_on_t04_1024",  True,  0.4, 1024, [1, 2, 3, 4, 5]),
    ("E_on_t0_2048",   True,  0.0, 2048, [0]),
]

# Leak = chars before the first structured clinical header (screen).
HEADER_RE = re.compile(
    r'(?im)^\s*(#+\s*)?(\*\*)?\s*(\d+\.\s*)?'
    r'(direct answer|patient summary|summary|diagnosis|assessment|clinical data|differential)'
)
def leak_chars(t):
    m = HEADER_RE.search(t)
    return m.start() if m else len(t)

# 7-item rubric, broadened patterns (screen only — real grade is reading the dumps).
RUBRIC = {
    "unifying_dx(HH)":  lambda t: ("hemochromatosis" in t) or ("iron overload" in t),
    "STOP_iron":        lambda t: (("ferrous" in t) or ("iron supplement" in t) or ("supplemental iron" in t))
                                   and any(w in t for w in ("stop", "discontinu", "contraindicat", "cease", "hold", "inappropriate", "avoid", "harmful")),
    "HFE_genotype":     lambda t: ("hfe" in t) or ("c282y" in t) or ("h63d" in t),
    "phlebotomy":       lambda t: ("phlebotomy" in t) or ("venesection" in t),
    "sec_hypogonadism": lambda t: ("hypogonad" in t and any(w in t for w in ("secondary", "pituitary", "central"))) or ("pituitary iron" in t),
    "cardiac_iron":     lambda t: ("cardiomyopathy" in t) or ("cardiac iron" in t) or ("t2*" in t) or (("cardiac" in t or "myocard" in t) and "iron" in t),
    "PCT_ddx":          lambda t: "porphyria" in t,
}
def catches(t):
    low = t.lower()
    return {k: fn(low) for k, fn in RUBRIC.items()}

OUT = os.path.join(os.path.dirname(__file__), "ab_out")
os.makedirs(OUT, exist_ok=True)

agg = {}  # label -> dict of metric lists
for label, think, temp, mt, seeds in CELLS:
    agg[label] = {"wall": [], "chars": [], "leak": [], "catch": [], "think_seen": []}
    for s in seeds:
        mx.random.seed(s)
        t0 = time.time()
        text = _vision_generate(images_b64=[], prompt=PROMPT, temperature=temp,
                                max_tokens=mt, enable_thinking=think)
        wall = time.time() - t0
        c = catches(text)
        nhit = sum(c.values())
        lk = leak_chars(text)
        low = text.lower()
        think_seen = ("<think>" in low) or ("thinking process" in low) or ("understand user role" in low)
        agg[label]["wall"].append(wall)
        agg[label]["chars"].append(len(text))
        agg[label]["leak"].append(lk)
        agg[label]["catch"].append(nhit)
        agg[label]["think_seen"].append(think_seen)
        with open(os.path.join(OUT, f"{label}_seed{s}.txt"), "w") as f:
            f.write(text)
        print(f"{label} seed={s}  wall={wall:5.1f}s  chars={len(text):5d}  "
              f"leak={lk:5d}  catch={nhit}/7  think_seen={think_seen}  "
              f"hits={[k for k,v in c.items() if v]}")

def ms(xs):
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd

print("\n" + "=" * 78)
print(f"{'cell':16} {'think':5} {'temp':4} {'maxtok':6} {'n':2} "
      f"{'wall(s)':13} {'chars':13} {'leak':12} {'catch/7':11}")
for label, think, temp, mt, seeds in CELLS:
    a = agg[label]
    wm, ws = ms(a["wall"]); cm, cs = ms(a["chars"]); lm, ls = ms(a["leak"]); km, ks = ms(a["catch"])
    print(f"{label:16} {str(think):5} {temp:<4} {mt:<6} {len(seeds):<2} "
          f"{wm:5.1f}±{ws:4.1f}  {cm:5.0f}±{cs:4.0f}  {lm:5.0f}±{ls:4.0f}  {km:3.1f}±{ks:3.1f}")

print("\n--- key contrasts ---")
def cval(lbl, k): return statistics.mean(agg[lbl][k])
print(f"THINKING @ t0 (A vs B):   catch {cval('A_off_t0_1024','catch'):.0f} -> {cval('B_on_t0_1024','catch'):.0f}   "
      f"leak {cval('A_off_t0_1024','leak'):.0f} -> {cval('B_on_t0_1024','leak'):.0f}   "
      f"wall {cval('A_off_t0_1024','wall'):.1f} -> {cval('B_on_t0_1024','wall'):.1f}s")
print(f"THINKING @ t0.4 (C vs D): catch {cval('C_off_t04_1024','catch'):.1f} -> {cval('D_on_t04_1024','catch'):.1f}   "
      f"leak {cval('C_off_t04_1024','leak'):.0f} -> {cval('D_on_t04_1024','leak'):.0f}")
print(f"TEMPERATURE (A vs C):     catch {cval('A_off_t0_1024','catch'):.0f} -> {cval('C_off_t04_1024','catch'):.1f}")
print(f"CAP rescue (B vs E):      catch {cval('B_on_t0_1024','catch'):.0f} -> {cval('E_on_t0_2048','catch'):.0f}   "
      f"chars {cval('B_on_t0_1024','chars'):.0f} -> {cval('E_on_t0_2048','chars'):.0f}   "
      f"(does 2048 let think-on finish the answer?)")
print(f"\nfull outputs dumped to: {OUT}")
