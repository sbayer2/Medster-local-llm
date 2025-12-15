# Primitive functions for LLM-generated code
# These are the building blocks the agent can compose into custom analysis
#
# Efficiency Features:
# - Concurrent loading via load_patients_batch() - 8x faster for multi-patient analysis
# - Automatic caching of patient bundles and ID lists
# - Batch FHIR operations with built-in aggregation

from typing import List, Dict, Any, Optional
from pathlib import Path
from medster.tools.medical.api import (
    load_patient_bundle,
    list_available_patients,
    extract_conditions,
    extract_observations,
    extract_medications,
    # New batch/async operations
    load_multiple_patients_sync,
    batch_extract_conditions,
    batch_extract_observations,
    batch_extract_medications,
    batch_search_resources,
)
from medster.config import (
    COHERENT_DICOM_PATH_ABS,
    COHERENT_CSV_PATH_ABS,
    get_selected_model
)
from medster.utils.image_utils import (
    dicom_to_base64_png,
    load_ecg_image_from_csv,
    find_patient_dicom_files,
    scan_all_dicom_files,
    get_image_metadata,
    ImageConversionError
)


def load_patient(patient_id: str) -> Dict[str, Any]:
    """Load a patient's complete FHIR bundle."""
    bundle = load_patient_bundle(patient_id)
    return bundle if bundle else {}


def get_patients(limit: Optional[int] = None) -> List[str]:
    """Get list of available patient IDs."""
    return list_available_patients(limit=limit)


def search_resources(bundle: Dict, resource_type: str) -> List[Dict]:
    """Extract all resources of a given type from a FHIR bundle."""
    if not bundle:
        return []

    resources = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            resources.append(resource)
    return resources


def get_conditions(bundle: Dict) -> List[Dict]:
    """Extract condition/diagnosis data from a FHIR bundle."""
    return extract_conditions({"entry": [{"resource": r} for r in search_resources(bundle, "Condition")]})


def get_observations(bundle: Dict, category: Optional[str] = None) -> List[Dict]:
    """Extract observations (labs, vitals) from a FHIR bundle."""
    obs_bundle = {"entry": [{"resource": r} for r in search_resources(bundle, "Observation")]}
    observations = extract_observations(obs_bundle)

    if category:
        # Filter by FHIR category field (e.g., 'laboratory', 'vital-signs')
        filtered = []
        for obs in observations:
            obs_categories = obs.get("category", [])
            # Check if any of the observation's categories match the requested category
            if any(category.lower() == cat.lower() for cat in obs_categories):
                filtered.append(obs)
        return filtered
    return observations


def get_medications(bundle: Dict) -> List[Dict]:
    """Extract medication data from a FHIR bundle."""
    return extract_medications({"entry": [{"resource": r} for r in search_resources(bundle, "MedicationRequest")]})


def filter_by_text(items: List[Dict], field: str, search_text: str, case_sensitive: bool = False) -> List[Dict]:
    """Filter items where field contains search text."""
    results = []
    search = search_text if case_sensitive else search_text.lower()

    for item in items:
        value = str(item.get(field, ""))
        if not case_sensitive:
            value = value.lower()
        if search in value:
            results.append(item)
    return results


def filter_by_value(items: List[Dict], field: str, operator: str, threshold: float) -> List[Dict]:
    """Filter items by numeric comparison (gt, lt, gte, lte, eq)."""
    results = []
    for item in items:
        value = item.get(field)
        if value is None:
            continue
        try:
            num_value = float(value)
            if operator == "gt" and num_value > threshold:
                results.append(item)
            elif operator == "lt" and num_value < threshold:
                results.append(item)
            elif operator == "gte" and num_value >= threshold:
                results.append(item)
            elif operator == "lte" and num_value <= threshold:
                results.append(item)
            elif operator == "eq" and num_value == threshold:
                results.append(item)
        except (ValueError, TypeError):
            continue
    return results


def count_by_field(items: List[Dict], field: str) -> Dict[str, int]:
    """Count occurrences of each unique value in a field."""
    counts = {}
    for item in items:
        value = str(item.get(field, "Unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def group_by_field(items: List[Dict], field: str) -> Dict[str, List[Dict]]:
    """Group items by a field value."""
    groups = {}
    for item in items:
        key = str(item.get(field, "Unknown"))
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def aggregate_numeric(items: List[Dict], field: str) -> Dict[str, float]:
    """Calculate statistics for a numeric field."""
    values = []
    for item in items:
        val = item.get(field)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "sum": 0}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "sum": sum(values)
    }


####################################
# Batch Operations (High Efficiency)
####################################

def load_patients_batch(patient_ids: List[str]) -> Dict[str, Dict]:
    """
    Load multiple patient bundles concurrently.
    8x faster than sequential loading for large patient sets.

    Args:
        patient_ids: List of patient IDs to load

    Returns:
        Dict mapping patient_id -> FHIR bundle (or empty dict if not found)

    Example:
        patients = get_patients(100)
        bundles = load_patients_batch(patients)  # Loads all 100 concurrently
        for pid, bundle in bundles.items():
            if bundle:
                conditions = get_conditions(bundle)
    """
    result = load_multiple_patients_sync(patient_ids)
    # Convert None to empty dict for easier handling in generated code
    return {pid: (bundle or {}) for pid, bundle in result.items()}


def batch_conditions(
    patient_ids: List[str],
    condition_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and aggregate conditions from multiple patients in one call.
    Internally uses concurrent loading for maximum efficiency.

    Args:
        patient_ids: List of patient IDs to analyze
        condition_filter: Optional text filter (e.g., "diabetes", "hypertension")

    Returns:
        {
            "patients_analyzed": int,
            "patients_with_matches": int,
            "condition_counts": {condition_name: count},  # Sorted by frequency
            "patient_conditions": {patient_id: [conditions]}
        }

    Example:
        patients = get_patients(500)
        result = batch_conditions(patients, "diabetes")
        print(f"Found {result['patients_with_matches']} patients with diabetes")
        print(f"Top conditions: {list(result['condition_counts'].keys())[:5]}")
    """
    return batch_extract_conditions(patient_ids, condition_filter)


def batch_observations(
    patient_ids: List[str],
    category: Optional[str] = None,
    code_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and aggregate observations from multiple patients in one call.
    Includes automatic numeric statistics calculation.

    Args:
        patient_ids: List of patient IDs to analyze
        category: Optional FHIR category ('laboratory', 'vital-signs')
        code_filter: Optional text filter for observation codes

    Returns:
        {
            "patients_analyzed": int,
            "patients_with_data": int,
            "observation_counts": {code: count},
            "numeric_stats": {code: {"count", "min", "max", "mean"}},
            "patient_observations": {patient_id: [observations]}
        }

    Example:
        patients = get_patients(100)
        result = batch_observations(patients, category="laboratory", code_filter="glucose")
        if "Glucose" in result["numeric_stats"]:
            print(f"Average glucose: {result['numeric_stats']['Glucose']['mean']}")
    """
    return batch_extract_observations(patient_ids, category, code_filter)


def batch_medications(
    patient_ids: List[str],
    medication_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and aggregate medications from multiple patients in one call.

    Args:
        patient_ids: List of patient IDs to analyze
        medication_filter: Optional text filter for medication names

    Returns:
        {
            "patients_analyzed": int,
            "patients_with_medications": int,
            "medication_counts": {medication_name: count},
            "patient_medications": {patient_id: [medications]}
        }

    Example:
        patients = get_patients(200)
        result = batch_medications(patients, "metformin")
        print(f"{result['patients_with_medications']} patients on metformin")
    """
    return batch_extract_medications(patient_ids, medication_filter)


def batch_resources(
    patient_ids: List[str],
    resource_type: str,
    text_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for any FHIR resource type across multiple patients.
    Use this for resources without dedicated batch functions (AllergyIntolerance, Procedure, etc.)

    Args:
        patient_ids: List of patient IDs to search
        resource_type: FHIR resource type (e.g., 'AllergyIntolerance', 'Procedure', 'Immunization')
        text_filter: Optional text to filter resources (searches in resource text fields)

    Returns:
        {
            "resource_type": str,
            "patients_searched": int,
            "patients_with_results": int,
            "total_resources_found": int,
            "results": {patient_id: [resources]}
        }

    Example:
        patients = get_patients(100)
        allergies = batch_resources(patients, "AllergyIntolerance")
        print(f"Found allergies in {allergies['patients_with_results']} patients")
    """
    # Create filter function if text_filter provided
    filter_fn = None
    if text_filter:
        text_lower = text_filter.lower()
        def filter_fn(resource: dict) -> bool:
            # Search common text fields
            for field in ["code", "medicationCodeableConcept", "substance"]:
                obj = resource.get(field, {})
                if isinstance(obj, dict):
                    text = obj.get("text", "")
                    if text_lower in text.lower():
                        return True
                    for coding in obj.get("coding", []):
                        if text_lower in coding.get("display", "").lower():
                            return True
            return False

    return batch_search_resources(patient_ids, resource_type, filter_fn)


# Vision and Imaging Primitives

def scan_dicom_directory() -> List[str]:
    """
    Scan the DICOM directory and return all DICOM file paths.

    Returns:
        List of DICOM file paths as strings

    Example:
        dicom_files = scan_dicom_directory()
        for dicom_path in dicom_files[:10]:  # Sample first 10
            metadata = get_image_metadata(dicom_path)
            print(f"Modality: {{metadata.get('modality')}}")
    """
    try:
        dicom_files = scan_all_dicom_files(COHERENT_DICOM_PATH_ABS)
        return [str(f) for f in dicom_files]
    except Exception as e:
        return []


def find_patient_images(patient_id: str) -> Dict[str, Any]:
    """
    Find all available images for a patient (DICOM and ECG).

    Args:
        patient_id: Patient UUID (FHIR)

    Returns:
        Dictionary with 'dicom_files' (list of paths) and 'has_ecg' (bool)
    """
    try:
        # Load patient FHIR bundle to get demographics
        from medster.tools.medical.api import load_patient_bundle

        bundle = load_patient_bundle(patient_id)
        dicom_files = []

        if bundle:
            # Extract patient name from FHIR bundle
            patient_resources = [entry.get('resource') for entry in bundle.get('entry', [])
                                 if entry.get('resource', {}).get('resourceType') == 'Patient']

            if patient_resources:
                patient = patient_resources[0]
                names = patient.get('name', [])
                if names:
                    name_parts = names[0]
                    given = name_parts.get('given', [''])[0] if name_parts.get('given') else ''
                    family = name_parts.get('family', '')

                    # Match DICOM filename pattern: Given###_Family###_UUID...
                    # Try various pattern combinations
                    patterns = [
                        f"{given}*_{family}*",  # Full names
                        f"*{given}*{family}*",  # Name anywhere
                    ]

                    for pattern in patterns:
                        matched_files = list(COHERENT_DICOM_PATH_ABS.glob(f"{pattern}.dcm"))
                        if matched_files:
                            dicom_files = matched_files
                            break

        # Fallback: try UUID direct match (might work for some datasets)
        if not dicom_files:
            dicom_files = find_patient_dicom_files(COHERENT_DICOM_PATH_ABS, patient_id)

        # Check for ECG
        ecg_path = COHERENT_CSV_PATH_ABS / "observations.csv"
        has_ecg = False
        try:
            ecg_image = load_ecg_image_from_csv(ecg_path, patient_id)
            has_ecg = ecg_image is not None
        except Exception:
            pass

        return {
            "dicom_files": [str(f) for f in dicom_files],
            "dicom_count": len(dicom_files),
            "has_ecg": has_ecg
        }
    except Exception as e:
        return {"error": str(e), "dicom_files": [], "dicom_count": 0, "has_ecg": False}


def load_dicom_image(patient_id: str, image_index: int = 0) -> Optional[str]:
    """
    Load a DICOM image for a patient as optimized base64 PNG.

    Args:
        patient_id: Patient UUID
        image_index: Which image to load (0 for first, 1 for second, etc.)

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    try:
        dicom_files = find_patient_dicom_files(COHERENT_DICOM_PATH_ABS, patient_id)

        if not dicom_files or image_index >= len(dicom_files):
            return None

        image_path = dicom_files[image_index]
        base64_png = dicom_to_base64_png(image_path, target_size=(800, 800), quality=85)

        return base64_png

    except Exception as e:
        print(f"Error loading DICOM image: {e}")
        return None


def load_ecg_image(patient_id: str) -> Optional[str]:
    """
    Load ECG image for a patient from observations.csv.

    Args:
        patient_id: Patient UUID

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    try:
        ecg_path = COHERENT_CSV_PATH_ABS / "observations.csv"
        return load_ecg_image_from_csv(ecg_path, patient_id)
    except Exception as e:
        print(f"Error loading ECG image: {e}")
        return None


def get_dicom_metadata(patient_id: str, image_index: int = 0) -> Dict[str, Any]:
    """
    Get metadata for a patient's DICOM image.

    Args:
        patient_id: Patient UUID
        image_index: Which image to get metadata for

    Returns:
        Dictionary with modality, study description, dimensions, etc.
    """
    try:
        dicom_files = find_patient_dicom_files(COHERENT_DICOM_PATH_ABS, patient_id)

        if not dicom_files or image_index >= len(dicom_files):
            return {"error": "Image not found"}

        return get_image_metadata(dicom_files[image_index])

    except Exception as e:
        return {"error": str(e)}


def get_dicom_metadata_from_path(dicom_path: str) -> Dict[str, Any]:
    """
    Get metadata for a DICOM file from its path.

    Args:
        dicom_path: Full path to DICOM file (as string)

    Returns:
        Dictionary with modality, study description, dimensions, etc.

    Example:
        dicom_files = scan_dicom_directory()
        for path in dicom_files[:5]:
            metadata = get_dicom_metadata_from_path(path)
            print(f"Modality: {metadata.get('modality')}")
    """
    try:
        from pathlib import Path
        return get_image_metadata(Path(dicom_path))
    except Exception as e:
        return {"error": str(e)}


def analyze_image_with_llm(image_base64: str, prompt: str) -> str:
    """
    Analyze a medical image using the local vision model.

    This primitive enables autonomous vision analysis within generated code.
    Use this after loading images with load_dicom_image() or load_ecg_image().

    Args:
        image_base64: Base64-encoded PNG image string
        prompt: Clinical question or analysis request (e.g., "Does this ECG show atrial fibrillation?")

    Returns:
        Vision model analysis as text

    Example:
        ecg = load_ecg_image(patient_id)
        if ecg:
            analysis = analyze_image_with_llm(
                ecg,
                "Analyze this ECG for atrial fibrillation pattern. Report yes/no and key findings."
            )
    """
    try:
        from medster.model import call_llm

        response = call_llm(
            prompt=prompt,
            images=[image_base64],
            model=get_selected_model()  # Use selected vision model
        )

        # Extract text content from response
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"Vision analysis error: {str(e)}"


def analyze_ecg_for_rhythm(patient_id: str, clinical_context: str = "") -> Dict[str, Any]:
    """
    Analyze ECG image for cardiac rhythm with structured parsing.

    This primitive loads the ECG, performs vision analysis, and parses the result
    into structured data to avoid false positives from keyword matching.

    Args:
        patient_id: Patient UUID
        clinical_context: Optional clinical context (e.g., "Patient with HTN and Hyperlipidemia")

    Returns:
        Dictionary with structured rhythm analysis:
        {
            "patient_id": str,
            "ecg_available": bool,
            "rhythm": str (e.g., "Normal Sinus Rhythm", "Atrial Fibrillation", "Other"),
            "afib_detected": bool,
            "rr_intervals": str (e.g., "Regular", "Irregular", "Irregularly Irregular"),
            "p_waves": str (e.g., "Present and normal", "Absent", "Abnormal"),
            "baseline": str (e.g., "Normal", "Fibrillatory", "Other"),
            "confidence": str (e.g., "High", "Medium", "Low"),
            "clinical_significance": str,
            "raw_analysis": str
        }

    Example:
        result = analyze_ecg_for_rhythm("patient-uuid-123", "HTN + Hyperlipidemia")
        if result["afib_detected"]:
            print(f"AFib detected with {result['confidence']} confidence")
    """
    try:
        # Load ECG image
        ecg_image = load_ecg_image(patient_id)

        if not ecg_image:
            return {
                "patient_id": patient_id,
                "ecg_available": False,
                "rhythm": "Unknown",
                "afib_detected": False,
                "rr_intervals": "Unknown",
                "p_waves": "Unknown",
                "baseline": "Unknown",
                "confidence": "N/A",
                "clinical_significance": "No ECG image available for analysis",
                "raw_analysis": ""
            }

        # Structured prompt for ECG rhythm analysis
        context_str = f" (Clinical context: {clinical_context})" if clinical_context else ""
        prompt = f"""Analyze this ECG tracing for patient {patient_id}{context_str}.

Specifically assess for atrial fibrillation patterns and provide your analysis in this EXACT format:

RHYTHM: [State the rhythm - Normal Sinus Rhythm, Atrial Fibrillation, or Other]
R-R INTERVALS: [Regular, Irregular, or Irregularly Irregular]
P WAVES: [Present and normal, Absent, or Abnormal]
BASELINE: [Normal, Fibrillatory, or Other]
CLINICAL SIGNIFICANCE: [Brief clinical assessment]
CONFIDENCE: [High, Medium, or Low]

Be precise in your RHYTHM classification. Only state "Atrial Fibrillation" if you see irregularly irregular R-R intervals, absent P waves, AND fibrillatory baseline."""

        # Get vision analysis
        from medster.model import call_llm
        response = call_llm(
            prompt=prompt,
            images=[ecg_image],
            model=get_selected_model()  # Use selected vision model (qwen3-vl:8b or ministral-3:8b)
        )

        raw_text = response.content if hasattr(response, 'content') else str(response)

        # Parse structured response with better logic
        def extract_field(text: str, field_name: str) -> str:
            """Extract value after 'FIELD_NAME:' line"""
            import re
            pattern = rf'{field_name}:\s*(.+?)(?:\n|$)'
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Unknown"

        rhythm = extract_field(raw_text, "RHYTHM")
        rr_intervals = extract_field(raw_text, "R-R INTERVALS")
        p_waves = extract_field(raw_text, "P WAVES")
        baseline = extract_field(raw_text, "BASELINE")
        significance = extract_field(raw_text, "CLINICAL SIGNIFICANCE")
        confidence = extract_field(raw_text, "CONFIDENCE")

        # Determine AFib based on RHYTHM field, not keyword matching
        afib_detected = False
        rhythm_lower = rhythm.lower()

        if "atrial fibrillation" in rhythm_lower or rhythm_lower == "afib":
            afib_detected = True
        elif "normal sinus rhythm" in rhythm_lower or rhythm_lower == "nsr":
            afib_detected = False
        # Secondary check: if rhythm unclear, check for classic AFib triad
        elif rhythm_lower == "unknown" or rhythm_lower == "other":
            afib_triad = (
                "irregularly irregular" in rr_intervals.lower() and
                "absent" in p_waves.lower() and
                "fibrillatory" in baseline.lower()
            )
            afib_detected = afib_triad

        return {
            "patient_id": patient_id,
            "ecg_available": True,
            "rhythm": rhythm,
            "afib_detected": afib_detected,
            "rr_intervals": rr_intervals,
            "p_waves": p_waves,
            "baseline": baseline,
            "confidence": confidence,
            "clinical_significance": significance,
            "raw_analysis": raw_text
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "ecg_available": False,
            "rhythm": "Error",
            "afib_detected": False,
            "rr_intervals": "Error",
            "p_waves": "Error",
            "baseline": "Error",
            "confidence": "N/A",
            "clinical_significance": f"Analysis error: {str(e)}",
            "raw_analysis": ""
        }


def analyze_multiple_images_with_llm(images: List[str], prompt: str) -> str:
    """
    Analyze multiple medical images together using the local vision model.

    Use this to compare images or analyze them in context of each other.

    Args:
        images: List of base64-encoded PNG image strings
        prompt: Clinical question or analysis request

    Returns:
        Vision model analysis as text

    Example:
        images = [load_dicom_image(pid, 0) for pid in patient_ids]
        images = [img for img in images if img]  # Remove None values
        if images:
            analysis = analyze_multiple_images_with_llm(
                images,
                "Compare these brain MRIs and identify any masses or hemorrhage."
            )
    """
    try:
        from medster.model import call_llm

        # Filter out None values
        valid_images = [img for img in images if img]

        if not valid_images:
            return "No valid images to analyze"

        response = call_llm(
            prompt=prompt,
            images=valid_images,
            model=get_selected_model()  # Use selected vision model
        )

        # Extract text content from response
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"Vision analysis error: {str(e)}"


# API specification for LLM code generation
PRIMITIVES_SPEC = """
Available functions for custom analysis:

# Patient Data
get_patients(limit: int = None) -> List[str]
    # Returns list of patient IDs (cached after first call)

load_patient(patient_id: str) -> Dict
    # Returns complete FHIR bundle for a patient (cached)

# ========== HIGH-EFFICIENCY BATCH OPERATIONS ==========
# Use these for multi-patient analysis - 8x faster than loops!

load_patients_batch(patient_ids: List[str]) -> Dict[str, Dict]
    # Load multiple patient bundles CONCURRENTLY (8 parallel threads)
    # Returns: {{patient_id: bundle}} - empty dict if patient not found
    # Example:
    #   patients = get_patients(100)
    #   bundles = load_patients_batch(patients)  # Loads all 100 at once!
    #   for pid, bundle in bundles.items():
    #       conditions = get_conditions(bundle)

batch_conditions(patient_ids: List[str], condition_filter: str = None) -> Dict
    # Extract conditions from ALL patients in ONE call with aggregation
    # Returns: {{
    #   "patients_analyzed": int,
    #   "patients_with_matches": int,
    #   "condition_counts": {{condition_name: count}},  # Sorted by frequency
    #   "patient_conditions": {{patient_id: [conditions]}}
    # }}
    # Example:
    #   result = batch_conditions(get_patients(500), "diabetes")
    #   print(f"{{result['patients_with_matches']}} patients with diabetes")

batch_observations(patient_ids: List[str], category: str = None, code_filter: str = None) -> Dict
    # Extract observations with automatic numeric statistics
    # category: "laboratory", "vital-signs"
    # Returns: {{
    #   "patients_analyzed": int,
    #   "observation_counts": {{code: count}},
    #   "numeric_stats": {{code: {{"count", "min", "max", "mean"}}}},
    #   "patient_observations": {{patient_id: [observations]}}
    # }}
    # Example:
    #   result = batch_observations(patients, "laboratory", "glucose")
    #   print(f"Avg glucose: {{result['numeric_stats']['Glucose']['mean']}}")

batch_medications(patient_ids: List[str], medication_filter: str = None) -> Dict
    # Extract medications from ALL patients in ONE call
    # Returns: {{
    #   "patients_analyzed": int,
    #   "medication_counts": {{medication_name: count}},
    #   "patient_medications": {{patient_id: [medications]}}
    # }}

batch_resources(patient_ids: List[str], resource_type: str, text_filter: str = None) -> Dict
    # Search ANY FHIR resource type across multiple patients
    # resource_type: "AllergyIntolerance", "Procedure", "Immunization", etc.
    # Returns: {{
    #   "patients_searched": int,
    #   "patients_with_results": int,
    #   "total_resources_found": int,
    #   "results": {{patient_id: [resources]}}
    # }}
    # Example:
    #   allergies = batch_resources(patients, "AllergyIntolerance")

# ========== SINGLE-PATIENT OPERATIONS ==========
# Use for detailed analysis of individual patients

# Resource Extraction
search_resources(bundle: Dict, resource_type: str) -> List[Dict]
    # Extract resources by type: "Patient", "Condition", "Observation", "MedicationRequest"

get_conditions(bundle: Dict) -> List[Dict]
    # Returns: [{{"name": str, "code": str, "clinical_status": str, "category": list}}]

get_observations(bundle: Dict, category: str = None) -> List[Dict]
    # Returns: [{{"code": str, "value": any, "unit": str, "effectiveDateTime": str}}]
    # category: "laboratory", "vital-signs"

get_medications(bundle: Dict) -> List[Dict]
    # Returns: [{{"medication": str, "status": str, "dosageInstruction": str}}]

# Filtering
filter_by_text(items: List, field: str, search_text: str) -> List[Dict]
    # Filter where field contains text (case-insensitive)

filter_by_value(items: List, field: str, operator: str, threshold: float) -> List[Dict]
    # operator: "gt", "lt", "gte", "lte", "eq"

# Aggregation
count_by_field(items: List, field: str) -> Dict[str, int]
    # Count occurrences of each unique value

group_by_field(items: List, field: str) -> Dict[str, List]
    # Group items by field value

aggregate_numeric(items: List, field: str) -> Dict
    # Returns: {{"count", "min", "max", "mean", "sum"}}

# Vision and Imaging (Multimodal Analysis)
scan_dicom_directory() -> List[str]
    # Scan DICOM directory and return ALL DICOM file paths
    # Returns: List of file path strings
    # Use this for database-wide DICOM analysis (fast - no patient iteration)
    # Example: dicom_files = scan_dicom_directory()  # Returns all 298 files

get_dicom_metadata_from_path(dicom_path: str) -> Dict
    # Get metadata for DICOM file from file path
    # Returns: {"modality": str, "study_description": str, "body_part": str, "dimensions": str, ...}
    # Use with scan_dicom_directory() for fast metadata extraction
    # Example: metadata = get_dicom_metadata_from_path(dicom_files[0])

find_patient_images(patient_id: str) -> Dict
    # Returns: {"dicom_files": List[str], "dicom_count": int, "has_ecg": bool}
    # Find all available images for a patient

load_dicom_image(patient_id: str, image_index: int = 0) -> Optional[str]
    # Load DICOM image as optimized base64 PNG string
    # image_index: 0 for first image, 1 for second, etc.
    # Returns base64 string ready for vision analysis

load_ecg_image(patient_id: str) -> Optional[str]
    # Load ECG image as base64 PNG string from observations.csv
    # Returns base64 string ready for vision analysis

get_dicom_metadata(patient_id: str, image_index: int = 0) -> Dict
    # Returns: {"modality": str, "study_description": str, "body_part": str, "dimensions": str, ...}
    # Get DICOM metadata without loading pixel data (requires patient ID)

analyze_image_with_llm(image_base64: str, prompt: str) -> str
    # Analyze a single medical image using local vision model
    # image_base64: Base64 PNG string from load_dicom_image() or load_ecg_image()
    # prompt: Clinical question (e.g., "Does this ECG show atrial fibrillation?")
    # Returns: Vision analysis as text
    # Example: analysis = analyze_image_with_llm(ecg, "Detect AFib pattern")

analyze_ecg_for_rhythm(patient_id: str, clinical_context: str = "") -> Dict
    # RECOMMENDED FOR ECG RHYTHM ANALYSIS - Structured parsing prevents false positives
    # Loads ECG, performs vision analysis, and parses result into structured data
    # Returns: {"patient_id", "ecg_available", "rhythm", "afib_detected", "rr_intervals",
    #           "p_waves", "baseline", "confidence", "clinical_significance", "raw_analysis"}
    # rhythm: "Normal Sinus Rhythm", "Atrial Fibrillation", or "Other"
    # afib_detected: bool (based on RHYTHM field, not keyword matching)
    # Example: result = analyze_ecg_for_rhythm(pid, "HTN + Hyperlipidemia")
    #          if result["afib_detected"]: print(f"AFib: {result['confidence']} confidence")

analyze_multiple_images_with_llm(images: List[str], prompt: str) -> str
    # Analyze multiple images together using local vision model
    # images: List of base64 PNG strings
    # prompt: Clinical question for comparative analysis
    # Returns: Vision analysis as text
    # Example: analysis = analyze_multiple_images_with_llm([img1, img2], "Compare these MRIs")

# Progress Logging
log_progress(message: str) -> None
    # Log progress during long-running analysis
    # Use this to report status when iterating through many patients
    # Example: log_progress(f"Processing patient {i+1}/{total}")
"""
