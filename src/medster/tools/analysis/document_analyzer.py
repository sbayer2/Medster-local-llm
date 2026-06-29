"""
Local Document Analysis Tool

Replaces the deprecated MCP server with local Qwen3.6-MLX model for
specialist-level clinical document analysis.

Uses the same model that powers the entire Medster agent — no remote API calls,
no API keys, fully on-device via Apple Silicon.

Original MCP architecture (deprecated):
- Local Agent: Claude Sonnet 4.5 (Medster) - Orchestration, tool selection
- Remote Server: Claude Sonnet 4.5 (FastMCP) - Specialist analysis

New local architecture:
- Single model: Qwen3.6-35B-A3B-MLX - Orchestration + specialist analysis
- Runs entirely on Apple Silicon (Metal + Neural Engine)
- No network calls, no API costs, no latency
"""

from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field

from medster.model import call_llm
from medster.config import get_selected_model


# =============================================================================
# Synthetic data disclaimer (same as MCP client)
# =============================================================================

SYNTHETIC_DATA_DISCLAIMER = """
[DISCLAIMER: This is SYNTHETIC patient data from the Coherent Data Set (SYNTHEA).
This is NOT real patient data - no PHI or HIPAA concerns apply.
This data is generated for medical AI research and education purposes.
Source: https://synthea.mitre.org/downloads - Coherent Data Set]

"""


# =============================================================================
# Input Schema
# =============================================================================

class DocumentAnalysisInput(BaseModel):
    """Input schema for local document analysis."""

    note_text: str = Field(
        description="The clinical note text to analyze (SOAP note, discharge summary, consult note, lab report, etc.)"
    )
    analysis_type: Literal["basic", "comprehensive", "complicated"] = Field(
        default="comprehensive",
        description="Level of analysis: 'basic' for quick extraction of key data, 'comprehensive' for detailed multi-step clinical reasoning, 'complicated' for deep analysis with differential diagnosis and quality assurance"
    )
    clinical_context: Optional[str] = Field(
        default="",
        description="Optional additional clinical context (e.g., patient demographics, known conditions, reason for consult)"
    )


# =============================================================================
# Analysis Prompts
# =============================================================================

_BASIC_ANALYSIS_PROMPT = """You are a clinical document analysis specialist. Analyze the following clinical note and extract key information in a structured format.

{synthetic_disclaimer}

Clinical Note:
{note_text}

{context}

Provide a structured analysis with:
1. Patient demographics (if mentioned)
2. Chief complaint / reason for visit
3. Key clinical findings (labs, vitals, imaging)
4. Diagnoses / assessments
5. Medications
6. Plan / recommendations

Keep it concise. Focus on extraction, not deep reasoning."""


_COMPREHENSIVE_ANALYSIS_PROMPT = """You are a clinical document analysis specialist performing a comprehensive analysis.

{synthetic_disclaimer}

Clinical Note:
{note_text}

{context}

Perform a thorough clinical analysis:

1. **SUMMARY**: One-paragraph clinical summary
2. **TIMELINE**: Chronological reconstruction of events
3. **KEY FINDINGS**: All clinically significant findings with values and reference ranges
4. **DIAGNOSTIC ASSESSMENT**:
   - Primary diagnosis with supporting evidence
   - Differential diagnoses
   - Confidence level for each
5. **TREATMENT ASSESSMENT**:
   - Current medications and adequacy
   - Interventions and their rationale
   - Gaps in treatment
6. **RISK ASSESSMENT**:
   - Critical values flagged
   - Drug interactions identified
   - Risk factors noted
7. **RECOMMENDATIONS**:
   - Immediate actions needed
   - Follow-up items
   - Missing data that should be obtained
8. **CLINICAL REASONING**: Brief explanation of how findings connect to diagnoses"""


_COMPLICATED_ANALYSIS_PROMPT = """You are a senior clinical specialist performing a deep, multi-step analysis of this case.

{synthetic_disclaimer}

Clinical Note:
{note_text}

{context}

Perform a comprehensive, multi-step clinical reasoning analysis:

## STEP 1: DATA EXTRACTION
Extract ALL clinical data points systematically:
- Demographics, vitals, labs, imaging, medications, history

## STEP 2: CLINICAL SYNTHESIS
- Build a coherent clinical picture
- Identify patterns and relationships between findings
- Note temporal relationships

## STEP 3: DIAGNOSTIC REASONING
- Primary diagnosis with full supporting evidence
- Differential diagnoses ranked by likelihood
- For each differential: supporting evidence, contradictory evidence, confidence
- Missing data that would help narrow the differential

## STEP 4: TREATMENT ANALYSIS
- Evaluate current treatment adequacy
- Identify drug interactions, contraindications, adverse effects
- Suggest evidence-based treatment modifications
- Note any treatment gaps

## STEP 5: RISK STRATIFICATION
- Flag critical values immediately
- Assess short-term and long-term risks
- Identify red flags requiring urgent attention
- Calculate relevant clinical scores if applicable (Wells, CHA2DS2-VASc, CURB-65, etc.)

## STEP 6: QUALITY ASSURANCE
- Self-check: Are there any findings you may have missed?
- Alternative interpretations of ambiguous data
- What would change your assessment?
- What single piece of information would be most valuable next?

## FINAL RECOMMENDATIONS
Prioritized list of actions:
1. Immediate (within hours)
2. Short-term (within days)
3. Long-term (follow-up, prevention)"""


# =============================================================================
# Tool
# =============================================================================

@tool(args_schema=DocumentAnalysisInput)
def analyze_document(
    note_text: str,
    analysis_type: Literal["basic", "comprehensive", "complicated"] = "comprehensive",
    clinical_context: str = ""
) -> dict:
    """
    Analyzes clinical documents using the local Qwen3.6-MLX model.

    Performs specialist-level clinical document analysis entirely on-device.
    No remote API calls, no API keys needed.

    Analysis types:
    - basic: Quick extraction of key clinical data points
    - comprehensive: Detailed multi-step clinical reasoning with assessment
    - complicated: Deep analysis with differential diagnosis, treatment evaluation,
      risk stratification, and quality assurance self-check

    Useful for: SOAP notes, discharge summaries, lab interpretations,
    consult notes, radiology reports, and any clinical text.

    Architecture: Uses the same Qwen3.6-35B-MLX model that powers the entire
    Medster agent — no separate server needed.

    Args:
        note_text: The clinical note to analyze
        analysis_type: Level of analysis depth
        clinical_context: Optional additional patient context

    Returns:
        Dictionary with analysis results including:
        - status: "success" or "error"
        - analysis_type: The type of analysis performed
        - analysis: The full analysis text
        - processing_time: Approximate processing time in seconds
    """
    try:
        # Select the appropriate prompt based on analysis type
        if analysis_type == "basic":
            analysis_prompt = _BASIC_ANALYSIS_PROMPT
        elif analysis_type == "complicated":
            analysis_prompt = _COMPLICATED_ANALYSIS_PROMPT
        else:  # comprehensive
            analysis_prompt = _COMPREHENSIVE_ANALYSIS_PROMPT

        # Build the formatted prompt
        context_str = f"Additional Clinical Context: {clinical_context}" if clinical_context else ""
        formatted_prompt = analysis_prompt.format(
            synthetic_disclaimer=SYNTHETIC_DATA_DISCLAIMER,
            note_text=note_text,
            context=context_str
        )

        # Call the local model for analysis
        import time
        start_time = time.time()

        response = call_llm(
            prompt=formatted_prompt,
            model=get_selected_model(),
            system_prompt=None  # The prompt itself contains all guidance
        )

        elapsed = time.time() - start_time

        # Extract text content from response
        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "analysis": analysis_text,
            "processing_time": round(elapsed, 2),
            "source": "Local Qwen3.6-MLX Model",
            "note_length": len(note_text),
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": str(e),
            "source": "Local Qwen3.6-MLX Model",
            "recommendation": "Check model availability and try again",
        }
