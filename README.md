# Medster-local-LLM — Autonomous Clinical Case Analysis Agent

**A fully on-device clinical reasoning agent that runs a single 35B vision-language model — agent loop *and* medical image analysis — entirely on an Apple Silicon Mac. No cloud, no API keys, no Ollama in the hot path. Zero per-token cost.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black)](https://github.com/ml-explore/mlx)
[![Model](https://img.shields.io/badge/Qwen3.6--35B--A3B-OptiQ%204bit-orange)](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit)

Medster plans, acts, validates, and synthesizes its way through clinical questions over real FHIR/DICOM data — and it does it all on one local model: **Qwen3.6-35B-A3B (OptiQ 4-bit)** served via **mlx_vlm** on the Apple Neural Engine + Metal. Built and validated on a **MacBook Pro (M5 Pro, 64 GB unified memory)**.

---

## The evolution — why this is now genuinely powerful locally

Medster began as a cloud (Claude API) agent, then a cost-saving local fork running small text/vision models through **Ollama**. The current generation is a step change:

| | Old (Ollama era) | **Now (single-OptiQ)** |
|---|---|---|
| Models | gpt-oss:20b (text) + qwen3-vl:8b (vision), switched at startup | **One** model: Qwen3.6-35B-A3B OptiQ 4-bit |
| Agent loop | Ollama, HTTP, grammar-constrained JSON | **mlx_vlm on-device** (`call_opti_llm`), JSON via schema-hint + retry |
| Vision | small 8B vision model, slow | **same 35B VLM** reads DICOM brain MRI/CT + ECG tracings |
| Runtimes resident | two models (~45 GB) + Ollama server | **one ~24 GB process**, flat memory |
| Cloud / API | none (local) | none (local) |

**Why it took a 64 GB Apple Silicon machine:** a 35B Mixture-of-Experts VLM at 4-bit is ~24 GB resident. The Neural Engine + Metal (via Apple's MLX) make a model this size genuinely fast on a laptop — and the OptiQ quantization keeps it on-device with a real vision encoder, so the *same* model handles clinical reasoning and image reads. That's the unlock: you no longer trade text quality for vision, or run two models that fight for memory.

> **Hard-won lesson baked in:** running the agent loop on Ollama *and* a separate vision model pushed a 64 GB machine into 50 GB+ KV-cache spikes and swap. Collapsing to one OptiQ model via mlx_vlm holds memory flat. See [Architecture](#architecture).

---

## 🚀 Quick Start (Apple Silicon)

```bash
# 1. Clone (keep it OUT of iCloud-synced folders like ~/Desktop — use ~/projects)
git clone https://github.com/sbayer2/Medster-local-LLM.git ~/projects/Medster-local-LLM
cd ~/projects/Medster-local-LLM

# 2. Install Python deps (uv recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync

# 3. Get the model (~24 GB, cached in ~/.cache/huggingface — one time)
uv run hf download mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit

# 4. Get the Coherent Data Set (~9 GB, public, from AWS Open Data — fast & resumable)
curl -L -C - -o coherent.zip \
  "https://synthea-open-data.s3.amazonaws.com/coherent/coherent-11-07-2022.zip"
unzip -q coherent.zip -d coherent_data && rm coherent.zip

# 5. Configure paths
cp env.example .env
# set OPTI_ALL_MODE=true and the four COHERENT_* paths to ~/projects/.../coherent_data/{fhir,dicom,csv,dna}

# 6. Run
uv run medster-agent      # pick option 1 (OptiQ)
```

No `ollama serve`, no API key. The first query loads the model once (~2 s warm cache) and caches it for the session.

---

## What you'll see

```
1. OptiQ 4-bit (Qwen3.6-35B-A3B) — single model: agent loop + vision
   - Runs on-device via mlx_vlm — NOT Ollama (model id: qwen3.6:35b-mlx)

medster>> research hypertension prevalence and comorbidities in 200 patients
  ✓ plan → analyze (1 tool call) → synthesize
  → 36.0% prevalence (72/200); diabetes 70.8%, heart failure 26.4%, CKD 18.1% …
  (single OptiQ model, ~24 GB flat, no swap)

medster>> examine the EKG tracing for patient 008d8e1d-…
  → analyze_patient_ecg → OptiQ vision → Normal Sinus Rhythm, AFib ruled out, P waves present …

medster>> examine the brain MRI for patient a1e1261c-…
  → analyze_patient_dicom → OptiQ vision → no mass/hemorrhage/midline shift; mild cortical atrophy …
```

---

## Hardware requirements

| | Recommended | Notes |
|---|---|---|
| **Machine** | Apple Silicon Mac (M-series) | Built/validated on **M5 Pro** |
| **Unified memory** | **64 GB** | Model ~24 GB resident; headroom for KV cache + image decode |
| **Disk** | ~40 GB free | ~24 GB model (`~/.cache`) + ~15 GB extracted dataset |
| **OS** | macOS (Metal + Neural Engine) | MLX runs on Apple's GPU/ANE |
| **Python** | 3.10+ | managed by `uv` |

32 GB Macs can run it but will be tight under long conversations (KV cache grows with context); 64 GB is the comfortable target.

---

## Architecture

```
                         MEDSTER CLI  (uv run medster-agent)
                                │
                 plan → act → validate → synthesize   (Dexter-style loop)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                             ▼
  Coherent Data Set                       ONE LOCAL MODEL via mlx_vlm
  (local FHIR/DICOM/CSV/DNA)              Qwen3.6-35B-A3B OptiQ 4-bit (~24 GB)
   • 1,280 FHIR bundles                    • agent loop  (call_opti_llm)
   • 298 DICOM brain scans                 • document analysis
   • ECGs in observations.csv             • VISION: DICOM + ECG reads
   • 889 genomic CSVs                      (Apple Neural Engine + Metal)

        100% ON-DEVICE  ·  ONE MODEL  ·  NO OLLAMA / NO API  ·  FLAT ~24 GB
```

**Single-model routing (`OPTI_ALL_MODE=true`, the default).** Every loop call — planning, tool selection, validation, synthesis — and every vision call routes through the OptiQ model via `mlx_vlm` (`call_opti_llm` / `_vision_generate`). Ollama remains an optional fallback (`OPTI_ALL_MODE=false`) but is not needed.

**Thinking & temperature, tuned per call:**
- **Agent loop + planning + synthesis: thinking OFF, temperature 0** — deterministic, clean JSON. (OptiQ emits chain-of-thought *inline* rather than in strippable `<think>` tags, so thinking-on would pollute structured/user-facing output; and planning *must* be thinking-off so the `TaskList` JSON parses.)
- **`complicated` document analysis: thinking ON, temperature 0.4** — differential-diagnosis depth where the reasoning helps.
- **`skip_arg_optimization`** drops a per-tool LLM round-trip (~⅓ fewer loop calls).

**Deterministic termination.** Per-task step budget, loop-detection by *tool name* (robust to arg variation), and "budget-exhausted → advance" guards prevent runaway loops.

**Memory model.** One ~24 GB process; KV cache is per-call/transient, so the footprint stays flat instead of the dual-model spike that previously tipped into swap.

---

## Multimodal: first-class, single-tool image reads

The same OptiQ model that runs the agent loop reads medical images. Two tools take a patient ID and handle loading + the vision call internally — no code generation, no base64 plumbing:

- **`analyze_patient_ecg(patient_id)`** — loads the patient's ECG (base64 PNG in `observations.csv`, LOINC 29303009) and returns a structured rhythm read (rhythm, AFib detection, R-R regularity, P waves). *Validated: Normal Sinus Rhythm read with correct AFib rule-out.*
- **`analyze_patient_dicom(patient_id)`** — loads the patient's brain MRI/CT (matched **by filename**, since the Coherent DICOM tag `PatientID` is an unrelated `SUBJECT####` value) and returns a structured radiology read (masses, hemorrhage, midline shift, ventricles, atrophy, white-matter). *Validated: correct "no mass/hemorrhage/shift, mild cortical atrophy" read.*

Both run the OptiQ vision encoder via mlx_vlm in seconds. For bespoke imaging logic, `generate_and_run_analysis` exposes the underlying primitives (`find_patient_images`, `load_dicom_image`, `load_ecg_image`, `analyze_image_with_llm`).

---

## Tools

**Medical data (SYNTHEA/FHIR):** `get_patient_labs`, `get_vital_signs`, `get_demographics`, `get_patient_conditions`, `get_clinical_notes`, `get_soap_notes`, `get_discharge_summary`, `get_medication_list`, `check_drug_interactions`, `get_radiology_reports`, `analyze_batch_conditions`

**Clinical scores:** `calculate_clinical_score` — Wells', CHA₂DS₂-VASc, CURB-65, MELD, …

**Vision (on-device OptiQ):** `analyze_patient_ecg`, `analyze_patient_dicom`, `analyze_medical_images`

**Document analysis:** `analyze_document` — basic / comprehensive / `complicated` (differential + QA), on the local model

**Dynamic code:** `generate_and_run_analysis` — sandboxed Python over FHIR + imaging primitives for cohort/population queries with custom logic

---

## Data: the Coherent Data Set

A 9 GB synthetic record set that links FHIR, DICOM, ECG, and genomic data — **no PHI, no HIPAA concerns**. Fastest source is **AWS Open Data** (public S3, no account):

```bash
curl -L -C - -o coherent.zip \
  "https://synthea-open-data.s3.amazonaws.com/coherent/coherent-11-07-2022.zip"
unzip -q coherent.zip -d coherent_data
```

Extracts to `coherent_data/{fhir,dicom,csv,dna}`: **1,280** FHIR bundles, **298** DICOM brain scans, ECG waveforms in `observations.csv`, **889** genomic CSVs. Point the four `COHERENT_*` paths in `.env` at these folders (absolute paths recommended).

> **macOS tip:** keep the repo *and* the dataset out of iCloud-synced folders (`~/Desktop`, `~/Documents`). iCloud "Optimize Storage" can offload the 9 GB dataset to cloud stubs and break local reads. Use `~/projects` or `~/data`.

> **Citation:** Walonoski J, et al. *The "Coherent Data Set": Combining Patient Data and Imaging in a Comprehensive, Synthetic Health Record.* Electronics. 2022; 11(8):1199. https://doi.org/10.3390/electronics11081199

---

## Example queries

```
# Population / cohort (Medster's sweet spot)
research hypertension prevalence and comorbidities in 200 patients
what is the prevalence of chronic kidney disease across 500 patients

# Single-patient clinical analysis
analyze patient aee205cf for cardiovascular risk — conditions, labs, vitals, meds, risk stratification

# Vision (single tool, on-device OptiQ)
examine the EKG tracing for patient 008d8e1d-c53f-47ac-fc06-aeb02ea5e3ec
examine the brain MRI for patient a1e1261c-a48c-42f7-ec01-10ef16340c0e and report imaging findings
```

---

## Configuration (`.env`)

```bash
# Routing — single OptiQ model for agent loop + vision (no Ollama needed)
OPTI_ALL_MODE=true

# Coherent Data Set (absolute paths; keep OUT of iCloud-synced folders)
COHERENT_DATA_PATH=/Users/you/projects/Medster-local-LLM/coherent_data/fhir
COHERENT_DICOM_PATH=/Users/you/projects/Medster-local-LLM/coherent_data/dicom
COHERENT_CSV_PATH=/Users/you/projects/Medster-local-LLM/coherent_data/csv
COHERENT_DNA_PATH=/Users/you/projects/Medster-local-LLM/coherent_data/dna

# Agent loop bounds
MAX_STEPS=20
MAX_STEPS_PER_TASK=5

# Optional fallback: set false to route text through Ollama instead of OptiQ
# OPTI_ALL_MODE=false   (then OLLAMA_MODEL / OLLAMA_BASE_URL apply)
```

Optional override: `VISION_MODEL_PATH` / `OMLX_*` are read if set; defaults point at the cached OptiQ snapshot.

---

## Web UI (optional)

A Next.js + FastAPI front end is included (`frontend/`, `run_dev.sh`) for chat-style interaction with live agent-phase status. The CLI is the primary, fully-validated interface; the Web UI is convenience tooling.

---

## Code structure

```
src/medster/
  agent.py                  # plan → act → validate → synthesize; _llm router (OptiQ/Ollama)
  model.py                  # call_opti_llm (mlx_vlm), call_llm (Ollama fallback), JSON parsing
  model_capabilities.py     # per-model flags (skip_arg_optimization, vision, …)
  config.py                 # OPTI_ALL_MODE, model + data paths
  prompts.py                # compositional BASE + MODEL_SPECIFIC + VISION_ADDON prompts
  tools/
    medical/                # FHIR: labs, vitals, conditions, notes, meds, imaging
    clinical/               # clinical scores
    analysis/
      primitives.py         # _vision_generate (OptiQ mlx_vlm), DICOM/ECG/FHIR primitives
      vision_analyzer.py    # analyze_patient_ecg, analyze_patient_dicom, analyze_medical_images
      document_analyzer.py  # analyze_document (basic/comprehensive/complicated)
      code_generator.py     # generate_and_run_analysis (sandboxed Python)
  utils/                    # context management, intro banner, logging
```

---

## Safety & disclaimer

**For research and education only.** Not for clinical decision-making without physician review. Critical-value flagging and drug-interaction checks are simplified, not production-grade. The Coherent data is synthetic. Always verify with appropriate clinical resources.

---

## Acknowledgments

- [Dexter](https://github.com/virattt/dexter) by @virattt — the plan→act→validate→synthesize loop Medster adapts for medicine
- [MLX](https://github.com/ml-explore/mlx) & [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Apple Silicon inference for the OptiQ VLM
- [Qwen3.6 / OptiQ 4-bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit) — the on-device vision-language model
- [SYNTHEA](https://synthetichealth.github.io/synthea/) & the [Coherent Data Set](https://synthea.mitre.org/downloads) — synthetic multimodal medical data
- [Medster](https://github.com/sbayer2/Medster) — the original Claude-powered agent this project descends from

## License

MIT
