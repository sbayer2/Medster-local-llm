# Medster — Architecture Decision Cycle (ADC)

Decisions captured in order: **context → decision → consequences**. Mark open vs done.

---

## ADC-001 — `analyze_document` "complicated": thinking OFF, temp 0, max_tokens 4096  · DONE (2026-07-01)

**Context.** The `complicated` level shipped with `enable_thinking=True`, `temperature=0.4`,
and the `_vision_generate` default `max_tokens=1024` (never overridden). A factorial
isolation test (`tests/thinking_ab_test.py` + `thinking_ab_cellF.py` + `thinking_cellG_4096.py`)
held the prompt fixed and gridded **thinking × temperature × token-budget** on a synthetic
hemochromatosis case (deterministic at temp 0), graded against a 7-item clinical rubric with
full-output eyeballing:

- `max_tokens=1024` **truncated every** complicated analysis mid-report, before the
  recommendations section where STOP-iron / HFE / phlebotomy live → catch ≤ 3/7.
- `enable_thinking=True` made this OptiQ (Qwen3.6-35B-A3B) model emit **pure chain-of-thought
  and never a structured answer** (`reached_answer=0%` at 1024 *and* 2048), with **no accuracy
  gain**: at 2048, think-on = reasoning dump 6/7, think-off = clean structured 6/7.
- **think-off + temp 0 + 4096** = **7/7, complete, clean ending, ~3.4k tokens, deterministic** —
  full differential table (correctly reasons down PCT and Addison's), ferrous sulfate flagged
  contraindicated, therapeutic phlebotomy, HFE genotyping, HCC + first-degree-relative screening.

**Decision.** For `complicated`: `enable_thinking=False`, `temperature=0`, `max_tokens=4096`
(new `_ANALYSIS_MAXTOK` map). `basic`/`comprehensive` unchanged (still 1024). Fix applies to the
OptiQ path (`OPTI_ALL_MODE=True`, the shipped config); the inactive Ollama fallback carries a `TODO`.

**Consequences.**
- (+) complicated analyses now complete (no silent truncation), deterministic, cleanly formatted.
- (−) latency ~47s at 4096 vs ~14s for the old (truncated, unusable) output.
- Validated on **one** synthetic case — `tests/thinking_ab_test.py` is the regression fixture;
  re-run on any model/hardware swap. Generalization to other document types is unverified.
- The premise "complicated needs thinking for the differential" was **falsified**: think-off
  reasons at least as well *inside* the structured format.
- Meta: the decision came from a small eval-harness that also surfaced the shipping bug (1024 cap)
  and exposed its own brittle metrics (leak regex, keyword grader) — treat evals as the sensor,
  not QA bolted on the side.
