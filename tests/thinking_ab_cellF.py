"""Cell F (think-OFF, temp 0, max_tokens 2048) — isolates thinking from budget by
pairing against cell E (think-ON, temp 0, 2048). If F >= E, think-off wins (same
answer, no wasted reasoning tokens); if F < E, thinking adds value at adequate budget.

Also fixes the leak metric: preamble = chars before the first markdown header
(^#{1,6}\\s). If NO header exists, the whole output is preamble = the model produced
pure reasoning and never emitted a structured answer. Re-applied to every dump in
tests/ab_out/ (no re-running of A-E; their outputs are deterministic/saved).

    uv run python tests/thinking_ab_cellF.py
"""
import os
import re
import glob
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
OUT = os.path.join(os.path.dirname(__file__), "ab_out")

# --- run cell F: think OFF, temp 0, 2048 (1 inference) ---
mx.random.seed(0)
t0 = time.time()
textF = _vision_generate(images_b64=[], prompt=PROMPT, temperature=0.0,
                         max_tokens=2048, enable_thinking=False)
wallF = time.time() - t0
with open(os.path.join(OUT, "F_off_t0_2048_seed0.txt"), "w") as f:
    f.write(textF)
print(f"cell F (off/t0/2048) ran: wall={wallF:.1f}s  chars={len(textF)}")

# --- corrected preamble/leak metric ---
HDR = re.compile(r'(?m)^#{1,6}\s')
def preamble(t):
    m = HDR.search(t)
    return m.start() if m else len(t)   # no header => entire output is reasoning
def reached_answer(t):
    return HDR.search(t) is not None

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
def catch(t):
    low = t.lower()
    return sum(fn(low) for fn in RUBRIC.values())

# --- re-grade all dumps with corrected metrics ---
rows = {}
for path in glob.glob(os.path.join(OUT, "*.txt")):
    label = re.sub(r'_seed\d+$', '', os.path.basename(path)[:-4])
    t = open(path).read()
    r = rows.setdefault(label, {"chars": [], "pre": [], "reached": [], "catch": []})
    r["chars"].append(len(t))
    r["pre"].append(preamble(t))
    r["reached"].append(reached_answer(t))
    r["catch"].append(catch(t))

def ms(xs):
    m = statistics.mean(xs)
    return m, (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

print("\n" + "=" * 84)
print(f"{'cell':17}{'n':>2}  {'chars':>11}  {'preamble/leak':>14}  {'reached_ans':>11}  {'catch/7':>9}")
for label in sorted(rows):
    r = rows[label]
    cm, cs = ms(r["chars"]); pm, ps = ms(r["pre"]); km, ks = ms(r["catch"])
    frac = sum(r["reached"]) / len(r["reached"])
    print(f"{label:17}{len(r['chars']):>2}  {cm:6.0f}±{cs:<4.0f}  {pm:6.0f}±{ps:<4.0f}    {frac:>9.0%}  {km:4.1f}±{ks:<3.1f}")

def v(label, key):
    return statistics.mean(rows[label][key]) if label in rows else float("nan")

print("\n--- THE DECISION: thinking vs budget, both at temp 0 / 2048 tokens ---")
for lbl in ("E_on_t0_2048", "F_off_t0_2048"):
    print(f"{lbl:16}  catch={v(lbl,'catch'):.0f}/7   preamble={v(lbl,'pre'):.0f} chars   "
          f"reached_answer={'yes' if v(lbl,'reached') >= 0.5 else 'NO'}   chars={v(lbl,'chars'):.0f}")
print("\ninterpretation: F>=E => think-OFF wins (same/better, no wasted reasoning tokens).")
print("                F< E => thinking earns its place once the budget is adequate.")
