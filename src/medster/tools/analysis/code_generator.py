# Dynamic code generation tool for custom analysis
# Allows the agent to generate and execute Python code when existing tools are insufficient

from langchain.tools import tool
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import traceback
import logging
from datetime import datetime

import random as _random

from medster.tools.analysis.primitives import (
    # Core patient data
    get_patients,
    load_patient,
    search_resources,
    get_conditions,
    get_observations,
    get_medications,
    # Filtering and aggregation
    filter_by_text,
    filter_by_value,
    count_by_field,
    group_by_field,
    aggregate_numeric,
    # NEW: High-efficiency batch operations
    load_patients_batch,
    batch_conditions,
    batch_observations,
    batch_medications,
    batch_resources,
    # Vision/imaging
    scan_dicom_directory,
    get_dicom_metadata_from_path,
    load_dicom_image_from_path,  # NEW: Load DICOM by file path (use with scan_dicom_directory)
    find_patient_images,
    load_dicom_image,
    load_ecg_image,
    get_dicom_metadata,
    analyze_image_with_llm,
    analyze_ecg_for_rhythm,
    analyze_multiple_images_with_llm,
    # NEW: OCR and batch vision (Qwen3.6-MLX enhanced)
    ocr_extract_text,
    analyze_batch_images,
    PRIMITIVES_SPEC
)


####################################
# Input Schema
####################################

class CodeGenerationInput(BaseModel):
    analysis_description: str = Field(
        default="Custom FHIR/clinical analysis",
        description="Natural language description of the analysis to perform. Be specific about what data to collect and how to aggregate it."
    )
    code: str = Field(
        description=f"Python code to execute using these primitives:\n{PRIMITIVES_SPEC}\nThe code must define a function called 'analyze()' that returns a dict with results."
    )
    patient_limit: int = Field(
        default=50,
        description="Maximum number of patients to analyze (for performance)."
    )


####################################
# Sandbox Environment
####################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MEDSTER CODE EXEC - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_sandbox_globals(patient_limit: int) -> dict:
    """Create a restricted global namespace for code execution."""

    def log_progress(message: str):
        """Progress logging function available to generated code."""
        logger.info(message)

    return {
        # FHIR Data Primitives (Single Patient)
        "get_patients": lambda limit=patient_limit: get_patients(limit),
        "load_patient": load_patient,
        "search_resources": search_resources,
        "get_conditions": get_conditions,
        "get_observations": get_observations,
        "get_medications": get_medications,
        "filter_by_text": filter_by_text,
        "filter_by_value": filter_by_value,
        "count_by_field": count_by_field,
        "group_by_field": group_by_field,
        "aggregate_numeric": aggregate_numeric,

        # HIGH-EFFICIENCY BATCH OPERATIONS (8x faster for multi-patient)
        "load_patients_batch": load_patients_batch,
        "batch_conditions": batch_conditions,
        "batch_observations": batch_observations,
        "batch_medications": batch_medications,
        "batch_resources": batch_resources,

        # Vision/Imaging Primitives
        "scan_dicom_directory": scan_dicom_directory,
        "get_dicom_metadata_from_path": get_dicom_metadata_from_path,
        "load_dicom_image_from_path": load_dicom_image_from_path,  # Load by path (use with scan_dicom_directory)
        "find_patient_images": find_patient_images,
        "load_dicom_image": load_dicom_image,  # Load by patient_id
        "load_ecg_image": load_ecg_image,
        "get_dicom_metadata": get_dicom_metadata,
        "analyze_image_with_llm": analyze_image_with_llm,
        "analyze_ecg_for_rhythm": analyze_ecg_for_rhythm,
        "analyze_multiple_images_with_llm": analyze_multiple_images_with_llm,
        # NEW: OCR and batch vision (Qwen3.6-MLX enhanced)
        "ocr_extract_text": ocr_extract_text,
        "analyze_batch_images": analyze_batch_images,

        # Safe built-ins
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "print": print,
        "hasattr": hasattr,
        "getattr": getattr,
        "isinstance": isinstance,
        "type": type,
        "ord": ord,  # For hash-based pseudo-random selection
        "chr": chr,  # Inverse of ord, useful for string operations

        # Random module for sampling
        "random": _random,

        # Typing module for type hints (avoid import errors)
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Optional": Optional,

        # Exception handling
        "Exception": Exception,

        # Progress logging
        "log_progress": log_progress,

        # No dangerous functions
        "__builtins__": {},
    }


####################################
# Tool
####################################

@tool(args_schema=CodeGenerationInput)
def generate_and_run_analysis(
    code: str,
    analysis_description: str = "Custom FHIR/clinical analysis",
    patient_limit: int = 50
) -> dict:
    """
    Generate & run custom Python over FHIR/vision primitives in a sandbox. Define
    analyze() returning a dict. NO import statements (primitives are pre-injected).

    SANDBOX API (use ONLY these names):
      get_patients(limit) -> List[str]
      load_patient(pid) -> bundle dict
      get_conditions(bundle) -> List[Dict]; each has keys: name, code (a STRING, not a dict), clinical_status
      get_observations(bundle), get_medications(bundle)
      search_resources(bundle, resource_type)  # 'Patient','Condition','AllergyIntolerance','Procedure',...
      batch_conditions(pids, filter), batch_observations(pids, category)
    IMAGING / ECG (match BY PATIENT, never by DICOM metadata):
      find_patient_images(pid) -> {dicom_count, has_ecg, dicom_files}
      load_dicom_image(pid, index=0) -> base64 PNG   # matched by FILENAME; DICOM tag PatientID is an unrelated SUBJECT#### value, do NOT compare it
      load_ecg_image(pid) -> base64 PNG              # ECGs live in observations.csv, NOT FHIR ImagingStudy
      analyze_image_with_llm(base64_png, prompt) -> str   # OptiQ vision read
    NOT sandbox primitives (do NOT call): get_patient_conditions, get_demographics.
      For demographics, read search_resources(bundle, 'Patient')[0].
    For a patient's brain MRI/CT, PREFER the analyze_patient_dicom(patient_id) tool;
    for an ECG, PREFER analyze_patient_ecg(patient_id) — no code needed.

    The code MUST define a function called 'analyze()' that returns a dict.

    Example code structure:
    ```
    def analyze():
        patients = get_patients(50)
        results = []
        for pid in patients:
            bundle = load_patient(pid)
            conditions = get_conditions(bundle)
            # ... custom analysis logic ...
        return {{"summary": results, "count": len(results)}}
    ```

    Example with autonomous vision analysis:
    ```
    def analyze():
        patients = get_patients(10)
        results = []
        for pid in patients:
            # Check for arrhythmia diagnosis
            bundle = load_patient(pid)
            conditions = get_conditions(bundle)
            has_arrhythmia = any('arrhythmia' in c.get('name', '').lower() for c in conditions)

            if has_arrhythmia:
                # Load and analyze ECG in one step
                ecg = load_ecg_image(pid)
                if ecg:
                    afib_analysis = analyze_image_with_llm(
                        ecg,
                        f"Analyze this ECG for patient {{pid}}. Does it show atrial fibrillation pattern? Answer yes or no with key findings."
                    )
                    results.append({{"patient": pid, "afib_analysis": afib_analysis}})
        return {{"afib_results": results}}
    ```

    Available primitives:
    - FHIR: get_patients, load_patient, get_conditions, get_observations, get_medications
    - Filtering: filter_by_text, filter_by_value
    - Aggregation: count_by_field, group_by_field, aggregate_numeric
    - Vision Loading: find_patient_images, load_dicom_image, load_ecg_image, get_dicom_metadata
    - Vision Analysis: analyze_image_with_llm, analyze_multiple_images_with_llm

    NOTE: You can now perform complete autonomous vision analysis within generated code!
    Use analyze_image_with_llm() to analyze images directly in your code without
    needing a separate tool call.
    """
    try:
        logger.info(f"Starting code execution: {analysis_description}")
        logger.info(f"Patient limit: {patient_limit}")

        # Create restricted sandbox
        sandbox_globals = create_sandbox_globals(patient_limit)
        sandbox_locals = {}

        # Strip the {{ }} template-escaping artifact that leaks from prompt/docstring
        # examples into generated code. The two LLM backends have opposite brace needs
        # (Ollama LangChain templates want {{ }}, OptiQ plain-string prompts want { }),
        # so doubled braces routinely appear in generated dict literals as invalid Python.
        # Generated analyze() code does not legitimately use {{ }}, so this is safe.
        code = code.replace('{{', '{').replace('}}', '}')

        # Execute the generated code
        exec(code, sandbox_globals, sandbox_locals)

        # Check if analyze function was defined
        if "analyze" not in sandbox_locals:
            return {
                "status": "error",
                "error": "Code must define a function called 'analyze()'",
                "description": analysis_description
            }

        # Run the analysis
        logger.info("Executing analyze() function...")
        start_time = datetime.now()
        result = sandbox_locals["analyze"]()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {elapsed:.2f} seconds")

        return {
            "status": "success",
            "description": analysis_description,
            "patient_limit": patient_limit,
            "result": result
        }

    except SyntaxError as e:
        return {
            "status": "error",
            "error": f"Syntax error in generated code: {str(e)}",
            "line": e.lineno,
            "description": analysis_description
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Execution error: {str(e)}",
            "traceback": traceback.format_exc(),
            "description": analysis_description
        }


# Export the primitives spec for the agent to reference
def get_primitives_spec() -> str:
    """Return the API specification for code generation."""
    return PRIMITIVES_SPEC
