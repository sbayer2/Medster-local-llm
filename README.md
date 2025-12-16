# Medster-local-LLM - Autonomous Clinical Case Analysis Agent

An autonomous agent for deep clinical case analysis powered by **local LLMs** - inspired by [Dexter](https://github.com/virattt/dexter) and adapted for medical domain with **zero API costs**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-brightgreen)](https://ollama.com)

**Choose Your Model at Startup:**
- **gpt-oss:20b** (text-only) - Fast clinical reasoning for labs, notes, reports
- **qwen3-vl:8b** (text + vision) - Multimodal analysis including DICOM images, ECG tracings, X-rays
- **ministral-3:8b** (text + vision) - Alternative vision model with strong reasoning

## 🚀 Quick Start

```bash
# 1. Install Ollama
brew install ollama  # macOS
# or download from https://ollama.com/download

# 2. Pull a local LLM model
ollama pull gpt-oss:20b  # Text-only (recommended to start)
# OR
ollama pull qwen3-vl:8b  # Text + Vision

# 3. Clone and setup
git clone https://github.com/sbayer/Medster-local-LLM.git
cd Medster-local-LLM
uv sync

# 4. Configure environment
cp env.example .env
# Edit .env to set paths (see below for beta testing)

# 5. Run CLI
uv run medster-agent

# OR run Web UI
./run_dev.sh
# Open http://localhost:3000
```

> **🧪 Beta Testers**: You need the Coherent Data Set (9GB) to test clinical analysis features.
> See [Beta Testing Setup](#-beta-testing-setup-coherent-data-set-required) below for complete database installation instructions.

## 🆕 Recent Changes

**December 2025:**

- **Compositional Prompts Framework** - Dynamic prompt composition with `BASE` + `MODEL_SPECIFIC` + `VISION_ADDON` layers for multi-model support
- **qwen3 Thinking Mode Fix** - Fixed JSON output parsing for models that use thinking mode (`think=False` binding)
- **Skip Arg Optimization** - New `skip_arg_optimization` flag for faster vision model execution
- **Async Primitives & Caching** - Added `async_load_patient()`, `@lru_cache` decorators, batch FHIR operations
- **Vision Keyword Detection** - Automatic detection of imaging queries to enable vision prompts
- **Two-Task DICOM Pattern** - Mandatory discovery + adaptation pattern for DICOM analysis
- **Frontend Updates** - Next.js 15.1.4, React 19.0.0
- **LangChain Template Fix** - Escaped curly braces in JSON examples to prevent template errors

**Model Testing Results:**

| Model | Planning | Action | Vision | Notes |
|-------|----------|--------|--------|-------|
| gpt-oss:20b | ✅ | ✅ | N/A | Fast, recommended for text |
| qwen3-vl:8b | ✅ | ✅ | ✅ | Slower (~3min/step) |
| ministral-3:8b | ✅ | ✅ | ✅ | Medium (~30s/step) |

## 🧪 Beta Testing Setup (Coherent Data Set Required)

**For beta testers**: To fully test Medster's clinical analysis capabilities, you need the Coherent Data Set with synthetic medical data.

### Step-by-Step Database Setup

**1. Download the Coherent Data Set**

Download the complete 9GB dataset from MITRE SYNTHEA:
- **Download link**: [https://synthea.mitre.org/downloads](https://synthea.mitre.org/downloads)
- **File name**: `coherent-11-07-2022.zip` (9GB compressed)
- **What's included**: FHIR patient records, DICOM images, ECG data, lab results, clinical notes

**2. Extract the Dataset**

```bash
# Navigate to your Medster-local-LLM directory
cd Medster-local-LLM

# Extract the downloaded zip file
unzip ~/Downloads/coherent-11-07-2022.zip

# Rename the extracted folder (if needed)
mv coherent coherent_data
```

**Expected directory structure after extraction:**
```
Medster-local-LLM/
├── coherent_data/
│   ├── fhir/           # 1,278 patient FHIR bundles (JSON files)
│   ├── dicom/          # 298 brain MRI scans (~9GB)
│   ├── csv/            # ECG waveforms and observations
│   └── dna/            # Genomic data (889 CSV files)
├── src/
├── frontend/
└── .env
```

**3. Configure Environment Variables**

Edit your `.env` file to point to the Coherent data directories:

```bash
# Open .env file
nano .env

# Add these paths (use absolute paths for reliability):
COHERENT_DATA_PATH=/Users/YOUR_USERNAME/Desktop/Medster-local-LLM/coherent_data/fhir
COHERENT_DICOM_PATH=/Users/YOUR_USERNAME/Desktop/Medster-local-LLM/coherent_data/dicom
COHERENT_CSV_PATH=/Users/YOUR_USERNAME/Desktop/Medster-local-LLM/coherent_data/csv
COHERENT_DNA_PATH=/Users/YOUR_USERNAME/Desktop/Medster-local-LLM/coherent_data/dna
```

**4. Verify Database Setup**

Run the verification script to confirm everything is configured correctly:

```bash
# Test FHIR data access
uv run python test_coherent_path.py

# Expected output:
# ✓ FHIR path exists: ./coherent_data/fhir
# ✓ Found 1278 patient bundles
# ✓ Sample patient loaded successfully
```

**5. Test Vision Capabilities (Optional)**

If using vision models (qwen3-vl:8b or ministral-3:8b), verify DICOM image access:

```bash
# Test DICOM image loading
uv run python test_vision.py

# Expected output:
# ✓ DICOM path exists: ./coherent_data/dicom
# ✓ Found 298 DICOM files
# ✓ Sample image loaded and converted successfully
```

### Beta Testing Queries

Once setup is complete, try these test queries:

**Text-Only Analysis (gpt-oss:20b):**
```
"Analyze patient demographics and find the top 5 most common conditions"
"Get lab results for patient 12345 and identify any critical values"
"Find patients with both hypertension AND diabetes"
```

**Vision Analysis (qwen3-vl:8b or ministral-3:8b):**
```
"Find patients with brain MRI scans and analyze the imaging findings"
"Analyze ECG waveforms for patients with atrial fibrillation"
"Review DICOM images for patients with stroke diagnosis"
```

### Troubleshooting Database Setup

**Issue**: `FileNotFoundError: coherent_data/fhir not found`

**Solution**:
1. Verify extraction: `ls -la coherent_data/fhir` should show JSON files
2. Use absolute paths in `.env` instead of relative paths
3. Check file permissions: `chmod -R 755 coherent_data`

**Issue**: `No patient files found`

**Solution**:
1. Confirm you downloaded the correct zip file (coherent-11-07-2022.zip)
2. Check FHIR folder contains .json files: `ls coherent_data/fhir/*.json | wc -l` should return 1278
3. Re-extract if necessary

**Issue**: DICOM images not loading

**Solution**:
1. Verify DICOM files exist: `ls coherent_data/dicom/*.dcm | wc -l` should return 298
2. Install pydicom: `uv add pydicom pillow`
3. Check COHERENT_DICOM_PATH in `.env` is correct

## Overview

Medster-local-LLM "thinks, plans, and learns as it works" - performing clinical analysis through task planning, self-reflection, and real-time medical data. It leverages SYNTHEA/FHIR data sources and runs **entirely on your local machine** using local LLMs via Ollama.

### Why Local LLM?

- **Cost Savings**: No API costs - runs 100% locally on your hardware
- **Privacy**: All medical data stays on your machine
- **Speed**: No network latency for API calls
- **Flexibility**: Works offline and supports any Ollama-compatible model
- **NEW**: Choose between text-only (faster) or multimodal vision (images) at startup
- **Dynamic Code Generation**: Agent writes custom Python code for complex analysis tasks

💡 **Model Options**: Select at startup between gpt-oss:20b (text-only, faster) or qwen3-vl:8b/ministral-3:8b (text + images for DICOM/ECG analysis) - see [Model Selection](#model-selection-text-only-vs-multimodal-vision)

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
    │          │  │  Models  │  │(optional)│
    │ FHIR     │  │          │  │ Complex  │
    │ Labs     │  │ Option 1:│  │ Analysis │
    │ Notes    │  │gpt-oss   │  │          │
    │ Reports  │  │  :20b    │  │          │
    │ DICOM    │  │ (TEXT)   │  │          │
    │ Images   │  │          │  │          │
    │          │  │ Option 2:│  │          │
    │          │  │qwen3-vl  │  │          │
    │          │  │  :8b     │  │          │
    │          │  │(VISION)  │  │          │
    └──────────┘  └──────────┘  └──────────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
     100% LOCAL - DUAL MODEL SUPPORT - NO API COSTS
```

## Requirements

- Python 3.10+
- **Ollama** installed locally (replaces Anthropic API)
- **Local LLM Model** (choose one or both):
  - **gpt-oss:20b** (text-only, 20B params, ~16GB RAM) - Recommended for fast reasoning
  - **qwen3-vl:8b** (multimodal vision, 8B params, ~6GB) - For image analysis
- Coherent Data Set with FHIR/DICOM data (optional but recommended)
- Optional: Your MCP medical analysis server for complex note analysis

### Hardware Requirements

**For gpt-oss:20b** (text-only, faster):
- **16GB+ RAM** or unified memory (Apple Silicon Macs)
- **CPU**: Modern multi-core processor
- **GPU** (optional but recommended): Speeds up inference significantly

**For qwen3-vl:8b** (multimodal vision):
- **8GB+ RAM** or unified memory
- **CPU/GPU**: Vision processing benefits from GPU acceleration
- Lower RAM requirements than gpt-oss:20b (~6GB model size)

**For gpt-oss:120b** (advanced users):
- **60GB+ VRAM** or unified memory
- Multi-GPU setup or high-end workstation

## 📦 Installation

### Prerequisites

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Ollama** - [Download](https://ollama.com/download)
- **uv** (recommended) or pip for dependency management
- **Node.js 18+** (for Web UI) - [Download](https://nodejs.org/)
- **16GB+ RAM** for gpt-oss:20b or **8GB+ RAM** for qwen3-vl:8b

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [https://ollama.com/download](https://ollama.com/download)

**Start Ollama service:**
```bash
ollama serve
# Leave this running in a terminal
```

### Step 2: Pull Local LLM Models

**Option 1: gpt-oss:20b (Text-Only, Recommended for Beginners)**
```bash
ollama pull gpt-oss:20b
```
- **Best for**: Lab analysis, clinical notes, medication reviews, text-based reasoning
- **Inference**: ~15-30 seconds per query
- **Model size**: ~16GB
- **RAM required**: 16GB+ (or unified memory on Apple Silicon)

**Option 2: qwen3-vl:8b (Multimodal Vision)**
```bash
ollama pull qwen3-vl:8b
```
- **Best for**: DICOM images, ECG tracings, X-rays, CT scans, MRI analysis
- **Inference**: Slower due to vision processing
- **Model size**: ~6GB
- **RAM required**: 8GB+

**Option 3: ministral-3:8b (Alternative Vision Model)**
```bash
ollama pull ministral-3:8b
```
- **Best for**: Balanced vision + reasoning capabilities
- **Model size**: ~8GB
- **RAM required**: 8GB+

**💡 Pro Tip:** Pull all three models and switch at runtime based on your analysis needs!

**Verify installation:**
```bash
ollama list
# Should show gpt-oss:20b, qwen3-vl:8b, and/or ministral-3:8b
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/sbayer/Medster-local-LLM.git
cd Medster-local-LLM
```

### Step 4: Install Python Dependencies

**Using uv (Recommended - Fast):**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

**Or using pip:**
```bash
pip install -e .
```

### Step 5: Configure Environment

```bash
cp env.example .env
```

**Edit `.env` file:**
```bash
# Ollama Configuration (NO API KEY NEEDED!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b  # or qwen3-vl:8b or ministral-3:8b

# Optional: Path to Coherent Data Set FHIR folder
# Leave blank to use agent without medical data
COHERENT_DATA_PATH=./coherent_data/fhir

# Optional: DICOM/CSV paths for vision analysis
COHERENT_DICOM_PATH=./coherent_data/dicom
COHERENT_CSV_PATH=./coherent_data/csv

# Optional: MCP medical analysis server
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_api_key_here
MCP_DEBUG=false  # Set to true for debugging
```

### Step 6: Download Coherent Data Set (Optional but Recommended)

The Coherent Data Set provides realistic synthetic medical data for testing.

**Download** (9GB): [https://synthea.mitre.org/downloads](https://synthea.mitre.org/downloads)

**Extract and configure:**
```bash
# After downloading and extracting:
unzip coherent-11-07-2022.zip
mv coherent coherent_data

# Update .env file:
COHERENT_DATA_PATH=./coherent_data/fhir
COHERENT_DICOM_PATH=./coherent_data/dicom
COHERENT_CSV_PATH=./coherent_data/csv
```

**What's included:**
- 1,278 FHIR patient bundles with longitudinal records
- 298 DICOM brain MRI scans (~9GB)
- ECG waveforms in CSV format
- Lab results, medications, conditions, procedures
- Clinical notes and discharge summaries

**✅ No API Keys Needed!** Everything runs locally on your machine.

### Step 7: Install Frontend Dependencies (Optional - for Web UI)

If you want to use the Web UI:

```bash
cd frontend
npm install
# or
yarn install
cd ..
```

## 🎯 Usage

### Web Interface (Recommended)

Medster-local-LLM includes a modern web interface built with Next.js and Tailwind CSS!

**Option 1: Quick Start (Automated)**
```bash
./run_dev.sh
```

This script automatically:
1. Starts FastAPI backend on `http://localhost:8000`
2. Installs frontend dependencies (if needed)
3. Starts Next.js frontend on `http://localhost:3000`

**Option 2: Manual Start**

**Terminal 1 - Backend:**
```bash
# Using uv
uv run uvicorn medster.api:app --reload --host 0.0.0.0 --port 8000

# Or using python directly
python -m uvicorn medster.api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# or
yarn dev
```

**Access the UI:**
Open your browser to `http://localhost:3000`

**Features:**
- 🎨 Premium medical-themed UI with glassmorphism effects
- 🔄 Real-time WebSocket streaming of agent responses
- 🤖 Dynamic model selection (gpt-oss:20b, qwen3-vl:8b, ministral-3:8b)
- 📊 Live status indicators showing agent planning, actions, and validation
- 💬 Chat-based interaction with your local medical AI
- 🔍 Task progress tracking with visual indicators
- ⚡ Markdown rendering with syntax highlighting

**API Documentation:**
Visit `http://localhost:8000/docs` for interactive FastAPI documentation (Swagger UI).

**API Endpoints:**
- `POST /chat` - WebSocket endpoint for streaming agent responses
- `POST /query` - REST endpoint for single queries
- `GET /models` - List available Ollama models
- `GET /health` - Health check endpoint

### Command Line Interface

Run the interactive CLI:
```bash
uv run medster-agent
```

Or:
```bash
python -m medster.cli
```

**CLI Features:**
- Interactive prompt with model selection
- Streaming agent responses
- Colored output for different agent phases
- Task progress tracking
- Exit with `/exit` or Ctrl+C

### Model Selection

At startup, you'll be prompted to choose your model:

```
======================================================================
MODEL SELECTION
======================================================================

Choose your model:

1. gpt-oss:20b (TEXT-ONLY)
   - Faster inference
   - Clinical reasoning, labs, notes, reports
   - Cannot process medical images

2. qwen3-vl:8b (TEXT + IMAGES)
   - Multimodal vision support
   - Can analyze DICOM images, ECG tracings, X-rays
   - Slower inference

======================================================================

Enter your choice (1 or 2):
```

**Choose Option 1 (gpt-oss:20b)** for:
- Text-based analysis (labs, reports, clinical notes)
- Faster inference (~15-30 seconds per query)
- Already downloaded and optimized for M4 Mac

**Choose Option 2 (qwen3-vl:8b)** for:
- Medical image analysis (X-rays, CT scans, DICOM files, ECG tracings)
- Multimodal vision capabilities
- Requires pulling the model first: `ollama pull qwen3-vl:8b`

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

## ⚙️ Configuration

### Environment Variables

All configuration is managed through the `.env` file:

```bash
# Ollama Configuration (NO API KEY NEEDED!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b  # Default model (changed at startup)

# Coherent Data Set Paths (Optional)
COHERENT_DATA_PATH=./coherent_data/fhir
COHERENT_DICOM_PATH=./coherent_data/dicom
COHERENT_CSV_PATH=./coherent_data/csv
COHERENT_DNA_PATH=./coherent_data/dna

# MCP Server (Optional - for complex note analysis)
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_api_key_here
MCP_DEBUG=false  # Enable detailed MCP logging

# Agent Configuration
MAX_STEPS=20  # Maximum total steps per query
MAX_STEPS_PER_TASK=5  # Maximum steps per individual task
```

### Using Different Models

You can use any Ollama-compatible model:

**Large Reasoning Models:**
```bash
OLLAMA_MODEL=gpt-oss:120b  # 120B params (requires 60GB+ RAM)
OLLAMA_MODEL=llama3.1:70b  # Meta's Llama 3.1 70B
OLLAMA_MODEL=qwen2.5:32b   # Qwen 2.5 32B
```

**Vision Models:**
```bash
OLLAMA_MODEL=qwen3-vl:8b      # Qwen3 Vision-Language
OLLAMA_MODEL=ministral-3:8b   # Ministral Vision
OLLAMA_MODEL=llava:13b        # LLaVA multimodal
```

**Note:** The agent will auto-detect model capabilities (text-only vs. vision) and adjust available tools accordingly.

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Connection refused to localhost:11434` | Run `ollama serve` to start Ollama |
| `model 'gpt-oss:20b' not found` | Run `ollama pull gpt-oss:20b` |
| Agent crashes / out of memory | Use smaller model (qwen3-vl:8b) or close other apps |
| WebSocket connection failed | Check backend: `curl http://localhost:8000/health` |
| `FileNotFoundError: coherent_data/fhir` | Use absolute paths in `.env` |
| Vision analysis not working | Ensure you selected qwen3-vl:8b or ministral-3:8b at startup |

**Note:** Most setup issues have been resolved. See [Beta Testing Setup](#-beta-testing-setup-coherent-data-set-required) for detailed database configuration.

## Safety & Disclaimer

**IMPORTANT**: Medster is for research and educational purposes only.

- Not intended for clinical decision-making without physician review
- Always verify findings with appropriate clinical resources
- Critical values and drug interactions are simplified checks
- Use clinical judgment for all patient care decisions

## 🏗️ Architecture Details

Medster implements a sophisticated multi-agent architecture with autonomous code generation and adaptive optimization:

### Core Agent Loop

1. **Planning Module** (`plan_tasks`)
   - Decomposes complex clinical queries into structured task sequences
   - Recognizes tool limitations (e.g., no allergy tool → use code generation)
   - Implements batch analysis optimization (single task vs. list + analyze)

2. **Action Module** (`ask_for_actions`)
   - Intelligent tool selection based on task context
   - **NEW:** Decision tree for dynamic code generation
   - Detects when to write custom Python code vs. use existing tools
   - Cross-task data access (all session outputs passed forward)

3. **Validation Module** (`ask_if_done`)
   - Single-task completion verification
   - **NEW:** Incomplete results detection (triggers data structure exploration)
   - Prevents false positives on "no data found" responses

4. **Synthesis Module** (`_generate_answer`)
   - Generates comprehensive clinical analysis
   - Structured output with mandatory sections
   - Token-efficient context management

### Advanced Features

**Dynamic Code Generation** (`generate_and_run_analysis`):
- Agent writes custom Python code for tasks without dedicated tools
- Sandboxed execution environment with FHIR primitives
- Use cases: allergies, procedures, immunizations, complex AND/OR logic
- Vision primitives for image analysis (DICOM, ECG waveforms)

**Adaptive Optimization**:
- Two-phase data discovery when results don't match expectations
- Phase 1: Explore actual data structure (sample DICOM metadata, etc.)
- Phase 2: Adapt code to match real data patterns
- Example: Discovers Coherent DICOM uses Modality='OT' (not textbook 'MR')

**Model Capability Registry** (`model_capabilities.py`):
- Auto-detects model capabilities (text-only vs. vision)
- Native tool calling vs. prompt-based tool selection
- Enables seamless model switching at startup

**Context Management** (`context_manager.py`):
- Token-efficient output truncation (max 50K tokens)
- Prioritizes recent outputs when context overflows
- Prevents runaway token usage with large datasets

### Safety Mechanisms

- **Global step limit**: 20 steps (configurable via `MAX_STEPS`)
- **Per-task step limit**: 5 steps (configurable via `MAX_STEPS_PER_TASK`)
- **Loop detection**: Prevents repetitive tool calls (tracks last 4 actions)
- **Error counter**: Max 3 consecutive errors before task completion
- **Tool execution tracking**: All outputs accumulated in `task_outputs` list
- **Critical value flagging**: Basic screening for abnormal lab values

## 📊 Differences from Original Medster

This is a **cost-saving fork** of the original [Medster](https://github.com/sbayer2/Medster) project:

| Feature | Original Medster | Medster-local-LLM |
|---------|------------------|-------------------|
| **LLM** | Claude Sonnet 4.5 (API) | gpt-oss:20b / qwen3-vl:8b / ministral-3:8b (Local) |
| **Modality** | Multimodal (text + images) | **Selectable at startup** - text or vision |
| **Image Support** | ✅ X-rays, CT, ECGs, documents | ✅ With qwen3-vl:8b or ministral-3:8b |
| **Cost** | ~$3-15 per 1M tokens | **$0 (100% local)** |
| **Privacy** | Data sent to Anthropic API | **All data stays local** |
| **Speed** | Fast (~2-5s API latency) | Text: fast / Vision: slower (local inference) |
| **Setup** | API key required | Ollama + model pull |
| **Hardware** | Any computer | 8-16GB+ RAM recommended |
| **Code Generation** | ❌ Fixed tools only | ✅ Dynamic Python code for custom tasks |
| **Adaptive Optimization** | ❌ | ✅ Data structure discovery + adaptation |
| **Model Flexibility** | Claude only | Any Ollama model (30+ options) |
| **Web UI** | ❌ | ✅ Next.js + FastAPI WebSocket streaming |

### Model Selection: Text-Only vs. Multimodal Vision

**NEW: Dual-model support!** Choose at startup between text-only and vision-capable models.

#### Option 1: gpt-oss:20b (TEXT-ONLY) ⚡

**Best for:**
- ✅ FHIR data, lab reports, clinical notes, vitals, medications, text-based ECG reports
- ✅ Faster inference (~15-30 seconds per query)
- ✅ Already optimized for M4 Mac (MXFP4 quantization)

**Cannot process:**
- ❌ DICOM images (X-rays, CT scans, MRIs)
- ❌ Image-based ECGs, scanned documents, handwritten notes

#### Option 2: qwen3-vl:8b (TEXT + IMAGES) 🖼️

**Best for:**
- ✅ All text-based analysis (same as gpt-oss:20b)
- ✅ **Medical image analysis**: X-rays, CT scans, DICOM files, ECG tracings
- ✅ Scanned documents and handwritten notes

**Trade-offs:**
- ⚠️ Slower inference (vision processing overhead)
- ⚠️ Requires model download: `ollama pull qwen3-vl:8b` (~4-5GB)

**Recommendation:** Start with gpt-oss:20b for text analysis, switch to qwen3-vl:8b when you need image analysis

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
