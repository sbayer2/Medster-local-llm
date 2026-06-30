# Medster-local-LLM — Autonomous Clinical Case Analysis Agent

**A fully on-device clinical reasoning agent: one 35B vision-language model handles both the analysis *and* the medical images — entirely on an Apple Silicon Mac. No cloud, no API keys, no patient data leaving the machine. Zero per-query cost.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black)](https://github.com/ml-explore/mlx)
[![Model](https://img.shields.io/badge/Qwen3.6--35B--A3B-OptiQ%204bit-orange)](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit)

Medster plans, acts, validates, and synthesizes its way through clinical questions over real FHIR/DICOM data — and it does it all on one local model: **Qwen3.6-35B-A3B (OptiQ 4-bit)** served via **mlx_vlm** on the Apple Neural Engine + Metal. Built and validated on a **MacBook Pro (M5 Pro, 64 GB unified memory)**.

---

## Why this matters — foundation-model clinical analysis, now on a desktop

Medster delivers accurate, autonomous clinical analysis over **large health datasets — patient records *and* medical images** — entirely on a physician's or researcher's local machine. Ask it to profile hypertension and its comorbidities across hundreds of patients, stratify a single case's cardiovascular risk, or read a brain MRI or ECG tracing — and it does so without a single byte of patient data leaving the desktop, and without a per-query cloud bill.

When we built the **original Medster in November 2025**, reaching this quality of multimodal clinical reasoning required large **hosted foundation models — API calls to Anthropic (Claude) and OpenAI (GPT)** — with the attendant per-token cost, network dependency, and patient data leaving the building. In the months since, open-model and Apple-Silicon progress changed the equation: a single 35B vision-language model, 4-bit quantized (**OptiQ**) and run through Apple's **MLX**, now produces **comparable analysis of FHIR records, lab and vital trends, ECG tracings, and DICOM brain scans — on a 64 GB MacBook Pro, fully offline.**

| | Original Medster (Nov 2025) | **Medster-local (now)** |
|---|---|---|
| Reasoning model | hosted Claude / GPT foundation models | Qwen3.6-35B-A3B OptiQ 4-bit, **on-device** |
| Where patient data goes | sent to Anthropic / OpenAI APIs | **never leaves the desktop** |
| Cost | per-token API billing | **$0 per query** |
| Connectivity | requires internet | **runs fully offline** |
| Images (DICOM / ECG) | cloud vision API | **same local model reads them** |
| Hardware | any machine + API key | Apple Silicon Mac, 64 GB |

For a clinician or research team, that means **PHI-safe, zero-marginal-cost, offline** analysis of whole patient cohorts and their imaging — the kind of work that a year ago meant shipping patient data to a cloud provider. The single-model design (one ~24 GB process serving both the reasoning loop and the vision encoder) is what makes that quality fit on a laptop; the technical details are in [Architecture](#architecture).

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
