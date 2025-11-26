# Medster-local-LLM - Autonomous Clinical Case Analysis Agent

An autonomous agent for deep clinical case analysis powered by **local LLM (gpt-oss:20b)** - inspired by [Dexter](https://github.com/virattt/dexter) and adapted for medical domain with **zero API costs**.

## Overview

Medster-local-LLM "thinks, plans, and learns as it works" - performing clinical analysis through task planning, self-reflection, and real-time medical data. It leverages SYNTHEA/FHIR data sources and runs **entirely on your local machine** using OpenAI's gpt-oss:20b model via Ollama.

### Why Local LLM?

- **Cost Savings**: No API costs - runs 100% locally on your hardware
- **Privacy**: All medical data stays on your machine
- **Speed**: No network latency for API calls
- **Flexibility**: Works offline and supports any Ollama-compatible model

⚠️ **Important**: gpt-oss:20b is **text-only** - it cannot process medical images like X-rays, CT scans, or DICOM files (see [limitations](#important-limitation-text-only-model))

## Core Capabilities

- **Intelligent Task Planning**: Breaks down complex clinical questions into structured diagnostic and therapeutic steps
- **Autonomous Execution**: Automatically selects and runs appropriate tools for data gathering (labs, notes, vitals, imaging, medications)
- **Self-Validation**: Verifies its own work and iterates until tasks are complete
- **Real-Time Medical Data**: Patient notes, lab results, vital sign trends, medication lists
- **Safety Mechanisms**: Loop detection, critical value flagging, drug interaction checking

## Primary Use Cases

1. **Clinical Case Analysis** - Comprehensive review of patient cases with risk stratification
2. **Differential Diagnosis Workup** - Prioritized differentials with optimal diagnostic sequences

## Architecture

```
                    MEDSTER CLI
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Coherent │  │  Ollama  │  │   MCP    │
    │ Data Set │  │  Local   │  │  Server  │
    │          │  │          │  │(optional)│
    │ FHIR     │  │gpt-oss   │  │ Complex  │
    │ Labs     │  │  :20b    │  │ Analysis │
    │ Notes    │  │TEXT-ONLY │  │          │
    │ Reports  │  │ Reasoning│  │          │
    └──────────┘  └──────────┘  └──────────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
        100% LOCAL - TEXT ONLY - NO API COSTS
```

## Requirements

- Python 3.10+
- **Ollama** installed locally (replaces Anthropic API)
- **gpt-oss:20b** model (20B parameters - requires ~16GB RAM/VRAM)
- FHIR server with patient data (default: HAPI FHIR test server)
- Optional: Your MCP medical analysis server for complex note analysis

### Hardware Requirements

For **gpt-oss:20b** (recommended):
- **16GB+ RAM** or unified memory (Apple Silicon Macs)
- **CPU**: Modern multi-core processor
- **GPU** (optional but recommended): Speeds up inference significantly

For **gpt-oss:120b** (advanced users):
- **60GB+ VRAM** or unified memory
- Multi-GPU setup or high-end workstation

## Installation

### Step 1: Install Ollama

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)

After installation, pull the gpt-oss:20b model:
```bash
ollama pull gpt-oss:20b
```

Verify installation:
```bash
ollama list
# Should show gpt-oss:20b in the list
```

### Step 2: Clone the Repository

```bash
git clone <your-repo-url>
cd Medster-local-LLM
```

### Step 3: Install Python Dependencies

Using uv (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

### Step 4: Configure Environment

```bash
cp env.example .env
```

Edit `.env` file with your configuration:
```bash
# Ollama Configuration (NO API KEY NEEDED!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Required: Path to Coherent Data Set FHIR folder
COHERENT_DATA_PATH=./coherent_data/fhir

# Optional: Your MCP medical analysis server
MCP_SERVER_URL=http://localhost:8000
```

### Step 5: Download Coherent Data Set (Optional but Recommended)

Download the 9GB Coherent Data Set from [https://synthea.mitre.org/downloads](https://synthea.mitre.org/downloads)

Extract and set the path in your `.env` file:
```bash
COHERENT_DATA_PATH=./coherent_data/fhir
```

**✅ No API Keys Needed!** Everything runs locally on your machine.

## Usage

Run the interactive CLI:
```bash
uv run medster-agent
```

Or:
```bash
python -m medster.cli
```

### Example Queries

**Clinical Case Analysis:**
```
medster>> Analyze this patient - 58yo male with chest pain, elevated troponins, and new ECG changes. What's the diagnostic workup and risk stratification?
```

**Lab Review:**
```
medster>> Get the last 7 days of labs for patient 12345 and identify any critical values or concerning trends
```

**Medication Safety:**
```
medster>> Review the medication list for patient 12345 and check for potential drug interactions
```

**Differential Diagnosis:**
```
medster>> Patient presents with fatigue, weight loss, and night sweats. Generate a prioritized differential and optimal workup sequence.
```

## Available Tools

### Medical Data (SYNTHEA/FHIR)
- `get_patient_labs` - Laboratory results with reference ranges
- `get_vital_signs` - Vital sign measurements and trends
- `get_demographics` - Patient demographic information
- `get_clinical_notes` - Progress notes, H&P, consultations
- `get_soap_notes` - SOAP-formatted progress notes
- `get_discharge_summary` - Hospital discharge summaries
- `get_medication_list` - Current and historical medications
- `check_drug_interactions` - Drug-drug interaction screening
- `get_radiology_reports` - Imaging studies and interpretations

### Clinical Scores
- `calculate_clinical_score` - Wells' Criteria, CHA2DS2-VASc, CURB-65, MELD, etc.

### Complex Analysis
- `analyze_complex_note` - Multi-step clinical reasoning via MCP server (Claude/Anthropic)

## Data Sources

### Coherent Data Set
Medster uses the **Coherent Data Set** - a comprehensive synthetic dataset that includes:
- FHIR resources (patient records, labs, vitals, medications, notes)
- DICOM images (X-rays, CT scans)
- Genomic data
- Physiological data (ECGs)
- Clinical notes

All data types are linked together via FHIR references.

**Download**: https://synthea.mitre.org/downloads (9 GB)

**Citation**:
> Walonoski J, et al. The "Coherent Data Set": Combining Patient Data and Imaging in a Comprehensive, Synthetic Health Record. Electronics. 2022; 11(8):1199. https://doi.org/10.3390/electronics11081199

### MCP Medical Analysis Server
For complex note analysis, Medster-local-LLM can optionally integrate with your FastMCP medical analysis server.

## Configuration

1. **Install Ollama** and pull gpt-oss:20b (see Installation section)
2. **Download and extract the Coherent Data Set** (optional but recommended)
3. **Set environment variables** in your `.env` file:

```bash
# Ollama (required - but NO API KEY!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Data paths
COHERENT_DATA_PATH=./coherent_data/fhir

# MCP server (optional)
MCP_SERVER_URL=http://localhost:8000
```

### Using Different Models

You can use any Ollama-compatible model by changing the `OLLAMA_MODEL` variable:

```bash
# For the larger 120B model (requires 60GB+ RAM)
OLLAMA_MODEL=gpt-oss:120b

# For other open models
OLLAMA_MODEL=llama3.1:70b
OLLAMA_MODEL=qwen2.5:32b
```

## Safety & Disclaimer

**IMPORTANT**: Medster is for research and educational purposes only.

- Not intended for clinical decision-making without physician review
- Always verify findings with appropriate clinical resources
- Critical values and drug interactions are simplified checks
- Use clinical judgment for all patient care decisions

## Architecture Details

Medster preserves Dexter's proven multi-agent architecture:

1. **Planning Module** - Decomposes clinical queries into tasks
2. **Action Module** - Selects appropriate tools for data retrieval
3. **Validation Module** - Verifies task completion
4. **Synthesis Module** - Generates comprehensive clinical analysis

Safety mechanisms include:
- Global step limits (default: 20)
- Per-task step limits (default: 5)
- Loop detection (prevents repetitive actions)
- Critical value flagging

## Differences from Original Medster

This is a **cost-saving fork** of the original [Medster](https://github.com/sbayer2/Medster) project:

| Feature | Original Medster | Medster-local-LLM |
|---------|------------------|-------------------|
| **LLM** | Claude Sonnet 4.5 (API) | gpt-oss:20b (Local) |
| **Modality** | **Multimodal** (text + images) | **Text-only** |
| **Image Support** | ✅ X-rays, CT scans, ECGs, documents | ❌ Requires preprocessing |
| **Cost** | ~$3-15 per 1M tokens | $0 (100% local) |
| **Privacy** | Data sent to Anthropic API | All data stays local |
| **Speed** | Network latency | Local inference |
| **Setup** | API key required | Ollama installation |
| **Hardware** | Any computer | 16GB+ RAM recommended |

### Important Limitation: Text-Only Model

**gpt-oss:20b is a text-only model** - it cannot directly process medical images. This is a key difference from Claude Sonnet 4.5's multimodal capabilities.

**What this means for medical data:**
- ✅ **Can process**: FHIR data, lab reports, clinical notes, vitals, medications, text-based ECG reports
- ❌ **Cannot process**: DICOM images (X-rays, CT scans, MRIs), image-based ECGs, scanned documents, handwritten notes

**Workarounds for image data:**
1. **Use radiology reports instead of images** - Extract text findings from DICOM metadata or reports
2. **OCR preprocessing** - Convert scanned documents to text before analysis
3. **Hybrid approach** - Use original Medster with Claude for image analysis, use Medster-local-LLM for text-based analysis
4. **Local vision models** - Consider multimodal Ollama models like `llama3.2-vision:11b` or `llava:13b` (experimental)

**When to use which version:**

- **Use Medster-local-LLM** if you want:
  - Zero API costs
  - Complete privacy
  - Offline capability
  - Local control
  - Text-based analysis only

- **Use Original Medster** if you want:
  - **Multimodal support** (images, PDFs, scans)
  - Latest Claude models
  - Fastest inference
  - No local hardware requirements
  - Cloud-based scaling

## License

MIT License

## Acknowledgments

- [Medster](https://github.com/sbayer2/Medster) - The original Claude-powered clinical analysis agent this project is forked from
- [Dexter](https://github.com/virattt/dexter) by @virattt - The original financial research agent that inspired this architecture. Medster adapts Dexter's proven multi-agent loop (planning → action → validation → synthesis) for the medical domain.
- [OpenAI gpt-oss](https://openai.com/open-models) - Open-source reasoning models that make local LLM deployment viable
- [Ollama](https://ollama.com) - Making local LLM deployment easy and accessible
- [SYNTHEA](https://synthetichealth.github.io/synthea/) - Synthetic patient data generator
- [HAPI FHIR](https://hapifhir.io/) - FHIR server implementation
- [Coherent Data Set](https://synthea.mitre.org/downloads) - 9GB synthetic dataset integrating FHIR, DICOM, genomics, and ECG data for comprehensive multimodal medical AI research
