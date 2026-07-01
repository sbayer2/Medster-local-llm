"""Cell G: think-OFF, temp 0, max_tokens 4096 — the winning config (F) with more
budget. Confirms whether the clipped 6/7 completes to 7/7 (adds phlebotomy +
monitoring), and sizes the right production max_tokens by checking if it finishes
naturally (clean ending) or still rides the cap.

    uv run python tests/thinking_cellG_4096.py
"""
import os
import re
import time
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

mx.random.seed(0)
t0 = time.time()
txt = _vision_generate(images_b64=[], prompt=PROMPT, temperature=0.0,
                       max_tokens=4096, enable_thinking=False)
wall = time.time() - t0
open(os.path.join(OUT, "G_off_t0_4096_seed0.txt"), "w").write(txt)

low = txt.lower()
hits = {k: fn(low) for k, fn in RUBRIC.items()}
reached = re.search(r'(?m)^#{1,6}\s', txt) is not None
# heuristic for "completed vs clipped": ended on sentence punctuation, not mid-word
tail = txt.rstrip()
clean_end = tail.endswith((".", "!", "?", ":", "*", "|", ")")) or tail.endswith("---")

print(f"G off/t0/4096: wall={wall:.1f}s  chars={len(txt)}  reached_answer={reached}  "
      f"catch={sum(hits.values())}/7  clean_end={clean_end}")
for k, v in hits.items():
    print(f"   [{'x' if v else ' '}] {k}")
print("\n--- last 400 chars (completed naturally, or still riding the 4096 cap?) ---")
print(txt[-400:])
