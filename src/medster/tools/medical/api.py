import os
import json
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from medster import config

####################################
# Coherent Data Set Configuration
####################################

# Use centralized config for Coherent Data Set path
# This ensures proper path resolution from .env file
COHERENT_DATA_PATH = str(config.COHERENT_FHIR_PATH_ABS)

# Thread pool for async I/O operations
_executor = ThreadPoolExecutor(max_workers=8)

# Cache for loaded patient data (simple dict cache)
_patient_cache: Dict[str, dict] = {}

# Cache for patient ID list (computed once)
_patient_list_cache: Optional[List[str]] = None


def load_patient_bundle(patient_id: str) -> Optional[dict]:
    """
    Load a patient's FHIR bundle from the Coherent Data Set.

    The Coherent Data Set stores each patient as a separate JSON bundle file.

    Args:
        patient_id: The patient's unique identifier

    Returns:
        dict: FHIR Bundle containing all patient resources, or None if not found
    """
    if patient_id in _patient_cache:
        return _patient_cache[patient_id]

    data_path = Path(COHERENT_DATA_PATH)

    # Try different file patterns used by Coherent Data Set
    patterns = [
        f"{patient_id}.json",
        f"*{patient_id}*.json",
        f"**/{patient_id}.json",
        f"**/*{patient_id}*.json",
    ]

    for pattern in patterns:
        matches = list(data_path.glob(pattern))
        if matches:
            with open(matches[0], 'r') as f:
                bundle = json.load(f)
                _patient_cache[patient_id] = bundle
                return bundle

    return None


def list_available_patients(limit: Optional[int] = None) -> List[str]:
    """
    List available patient IDs in the Coherent Data Set.
    Uses caching to avoid repeated filesystem scans.

    Args:
        limit: Maximum number of patients to return. None returns all patients.

    Returns:
        List of patient IDs
    """
    global _patient_list_cache

    # Return from cache if available
    if _patient_list_cache is not None:
        if limit is not None:
            return _patient_list_cache[:limit]
        return _patient_list_cache

    data_path = Path(COHERENT_DATA_PATH)
    if not data_path.exists():
        return []

    patient_ids = []
    for json_file in data_path.glob("**/*.json"):
        # Extract patient ID from filename or bundle
        try:
            with open(json_file, 'r') as f:
                bundle = json.load(f)
                # Find Patient resource in bundle
                for entry in bundle.get("entry", []):
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") == "Patient":
                        patient_ids.append(resource.get("id", json_file.stem))
                        break
                else:
                    patient_ids.append(json_file.stem)
        except:
            patient_ids.append(json_file.stem)

    # Cache the full list
    _patient_list_cache = patient_ids

    if limit is not None:
        return patient_ids[:limit]
    return patient_ids


####################################
# Async Operations for Concurrency
####################################

async def load_patient_bundle_async(patient_id: str) -> Optional[dict]:
    """
    Async version of load_patient_bundle for concurrent loading.

    Args:
        patient_id: The patient's unique identifier

    Returns:
        dict: FHIR Bundle containing all patient resources, or None if not found
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, load_patient_bundle, patient_id)


async def load_multiple_patients_async(patient_ids: List[str]) -> Dict[str, Optional[dict]]:
    """
    Load multiple patient bundles concurrently.

    Args:
        patient_ids: List of patient IDs to load

    Returns:
        Dict mapping patient_id -> bundle (or None if not found)
    """
    tasks = [load_patient_bundle_async(pid) for pid in patient_ids]
    results = await asyncio.gather(*tasks)
    return dict(zip(patient_ids, results))


def load_multiple_patients_sync(patient_ids: List[str]) -> Dict[str, Optional[dict]]:
    """
    Load multiple patient bundles concurrently (sync wrapper for async).
    Use this in non-async contexts like the sandbox.

    Args:
        patient_ids: List of patient IDs to load

    Returns:
        Dict mapping patient_id -> bundle (or None if not found)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, use thread pool directly
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(load_patient_bundle, patient_ids))
            return dict(zip(patient_ids, results))
        else:
            return loop.run_until_complete(load_multiple_patients_async(patient_ids))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(load_multiple_patients_async(patient_ids))


####################################
# Batch FHIR Operations
####################################

def batch_extract_conditions(patient_ids: List[str], condition_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract conditions from multiple patients in a single batch operation.

    Args:
        patient_ids: List of patient IDs to analyze
        condition_filter: Optional text filter for condition names (case-insensitive)

    Returns:
        Dict with aggregated condition data:
        {
            "patients_analyzed": int,
            "patients_with_matches": int,
            "condition_counts": {condition_name: count},
            "patient_conditions": {patient_id: [conditions]}
        }
    """
    # Load all bundles concurrently
    bundles = load_multiple_patients_sync(patient_ids)

    condition_counts: Dict[str, int] = {}
    patient_conditions: Dict[str, List[dict]] = {}
    patients_with_matches = 0

    for pid, bundle in bundles.items():
        if not bundle:
            continue

        conditions = extract_conditions(bundle)

        # Apply filter if specified
        if condition_filter:
            filter_lower = condition_filter.lower()
            conditions = [c for c in conditions if filter_lower in c.get("name", "").lower()]

        if conditions:
            patients_with_matches += 1
            patient_conditions[pid] = conditions

            for cond in conditions:
                name = cond.get("name", "Unknown")
                condition_counts[name] = condition_counts.get(name, 0) + 1

    # Sort condition counts by frequency
    sorted_counts = dict(sorted(condition_counts.items(), key=lambda x: x[1], reverse=True))

    return {
        "patients_analyzed": len(patient_ids),
        "patients_with_matches": patients_with_matches,
        "condition_counts": sorted_counts,
        "patient_conditions": patient_conditions
    }


def batch_extract_observations(
    patient_ids: List[str],
    category: Optional[str] = None,
    code_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract observations from multiple patients in a single batch operation.

    Args:
        patient_ids: List of patient IDs to analyze
        category: Optional FHIR category filter ('laboratory', 'vital-signs')
        code_filter: Optional text filter for observation codes

    Returns:
        Dict with aggregated observation data
    """
    bundles = load_multiple_patients_sync(patient_ids)

    observation_counts: Dict[str, int] = {}
    patient_observations: Dict[str, List[dict]] = {}
    numeric_values: Dict[str, List[float]] = {}  # For aggregation

    for pid, bundle in bundles.items():
        if not bundle:
            continue

        observations = extract_observations(bundle)

        # Apply category filter
        if category:
            observations = [o for o in observations if category.lower() in [c.lower() for c in o.get("category", [])]]

        # Apply code filter
        if code_filter:
            filter_lower = code_filter.lower()
            observations = [o for o in observations if filter_lower in o.get("code", "").lower()]

        if observations:
            patient_observations[pid] = observations

            for obs in observations:
                code = obs.get("code", "Unknown")
                observation_counts[code] = observation_counts.get(code, 0) + 1

                # Collect numeric values for aggregation
                value = obs.get("value")
                if isinstance(value, (int, float)):
                    if code not in numeric_values:
                        numeric_values[code] = []
                    numeric_values[code].append(float(value))

    # Calculate statistics for numeric observations
    numeric_stats: Dict[str, Dict[str, float]] = {}
    for code, values in numeric_values.items():
        if values:
            numeric_stats[code] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }

    return {
        "patients_analyzed": len(patient_ids),
        "patients_with_data": len(patient_observations),
        "observation_counts": dict(sorted(observation_counts.items(), key=lambda x: x[1], reverse=True)),
        "numeric_stats": numeric_stats,
        "patient_observations": patient_observations
    }


def batch_extract_medications(patient_ids: List[str], medication_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract medications from multiple patients in a single batch operation.

    Args:
        patient_ids: List of patient IDs to analyze
        medication_filter: Optional text filter for medication names

    Returns:
        Dict with aggregated medication data
    """
    bundles = load_multiple_patients_sync(patient_ids)

    medication_counts: Dict[str, int] = {}
    patient_medications: Dict[str, List[dict]] = {}

    for pid, bundle in bundles.items():
        if not bundle:
            continue

        medications = extract_medications(bundle)

        # Apply filter if specified
        if medication_filter:
            filter_lower = medication_filter.lower()
            medications = [m for m in medications if filter_lower in m.get("medication", "").lower()]

        if medications:
            patient_medications[pid] = medications

            for med in medications:
                name = med.get("medication", "Unknown")
                medication_counts[name] = medication_counts.get(name, 0) + 1

    return {
        "patients_analyzed": len(patient_ids),
        "patients_with_medications": len(patient_medications),
        "medication_counts": dict(sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)),
        "patient_medications": patient_medications
    }


def batch_search_resources(
    patient_ids: List[str],
    resource_type: str,
    filter_fn: Optional[Callable[[dict], bool]] = None
) -> Dict[str, Any]:
    """
    Search for any FHIR resource type across multiple patients.

    Args:
        patient_ids: List of patient IDs to search
        resource_type: FHIR resource type (e.g., 'AllergyIntolerance', 'Procedure', 'Immunization')
        filter_fn: Optional function to filter resources (receives resource dict, returns bool)

    Returns:
        Dict with search results per patient
    """
    bundles = load_multiple_patients_sync(patient_ids)

    results: Dict[str, List[dict]] = {}
    total_found = 0

    for pid, bundle in bundles.items():
        if not bundle:
            continue

        resources = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == resource_type:
                if filter_fn is None or filter_fn(resource):
                    resources.append(resource)

        if resources:
            results[pid] = resources
            total_found += len(resources)

    return {
        "resource_type": resource_type,
        "patients_searched": len(patient_ids),
        "patients_with_results": len(results),
        "total_resources_found": total_found,
        "results": results
    }


def clear_cache():
    """Clear all caches. Useful for testing or when data changes."""
    global _patient_cache, _patient_list_cache
    _patient_cache = {}
    _patient_list_cache = None


def search_fhir(resource_type: str, **search_params) -> dict:
    """
    Search for FHIR resources in a patient's bundle.

    Args:
        resource_type: FHIR resource type (Patient, Observation, etc.)
        **search_params: Search parameters including 'patient' ID

    Returns:
        dict: FHIR Bundle with search results
    """
    patient_id = search_params.get("patient", search_params.get("subject", ""))

    if not patient_id:
        return {"resourceType": "Bundle", "entry": [], "total": 0}

    bundle = load_patient_bundle(patient_id)
    if not bundle:
        return {
            "resourceType": "Bundle",
            "entry": [],
            "total": 0,
            "error": f"Patient {patient_id} not found in Coherent Data Set"
        }

    # Filter resources by type
    matching_entries = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            # Apply additional filters
            if _matches_search_params(resource, search_params):
                matching_entries.append(entry)

    # Apply limit
    limit = search_params.get("_count", 100)
    matching_entries = matching_entries[:limit]

    # Sort by date if requested
    sort_param = search_params.get("_sort", "")
    if sort_param:
        reverse = sort_param.startswith("-")
        sort_field = sort_param.lstrip("-")
        matching_entries = _sort_entries(matching_entries, sort_field, reverse)

    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(matching_entries),
        "entry": matching_entries
    }


def get_fhir_resource(resource_type: str, resource_id: str) -> dict:
    """
    Get a specific FHIR resource by ID.

    For Patient resources, the resource_id is the patient_id.
    For other resources, we need to search through patient bundles.

    Args:
        resource_type: FHIR resource type
        resource_id: Resource ID

    Returns:
        dict: FHIR Resource
    """
    if resource_type == "Patient":
        bundle = load_patient_bundle(resource_id)
        if bundle:
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    return resource

    # For other resources, we'd need to know which patient bundle to search
    # This is a limitation of file-based storage
    return {"error": f"Resource {resource_type}/{resource_id} not found"}


def _matches_search_params(resource: dict, params: dict) -> bool:
    """Check if a resource matches the search parameters."""
    # Category filter (e.g., 'laboratory', 'vital-signs')
    category = params.get("category", "")
    if category:
        resource_categories = resource.get("category", [])
        if isinstance(resource_categories, list):
            category_codes = []
            for cat in resource_categories:
                for coding in cat.get("coding", []):
                    category_codes.append(coding.get("code", "").lower())
            if category.lower() not in category_codes:
                return False

    # Code text filter
    code_text = params.get("code:text", "")
    if code_text:
        resource_code = resource.get("code", {}).get("text", "").lower()
        if code_text.lower() not in resource_code:
            return False

    # Status filter
    status = params.get("status", "")
    if status:
        if resource.get("status", "").lower() != status.lower():
            return False

    # Date filters (simplified)
    # In production, would parse and compare dates properly

    return True


def _sort_entries(entries: list, sort_field: str, reverse: bool) -> list:
    """Sort entries by a field."""
    def get_sort_key(entry):
        resource = entry.get("resource", {})
        # Common date fields
        for field in ["effectiveDateTime", "date", "issued", "authoredOn"]:
            if field in resource:
                return resource[field]
        return ""

    return sorted(entries, key=get_sort_key, reverse=reverse)


# Helper functions for common FHIR operations

def extract_observations(bundle: dict) -> list:
    """Extract observation data from a FHIR Bundle."""
    observations = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            # Extract category codes from FHIR structure
            category_codes = []
            for cat in resource.get("category", []):
                for coding in cat.get("coding", []):
                    code = coding.get("code", "")
                    if code:
                        category_codes.append(code)

            obs = {
                "code": resource.get("code", {}).get("text", "Unknown"),
                "value": None,
                "unit": None,
                "effectiveDateTime": resource.get("effectiveDateTime", ""),
                "status": resource.get("status", ""),
                "category": category_codes,  # e.g., ["vital-signs"] or ["laboratory"]
            }

            # Extract value (can be valueQuantity, valueString, etc.)
            if "valueQuantity" in resource:
                obs["value"] = resource["valueQuantity"].get("value")
                obs["unit"] = resource["valueQuantity"].get("unit", "")
            elif "valueString" in resource:
                obs["value"] = resource["valueString"]
            elif "valueCodeableConcept" in resource:
                obs["value"] = resource["valueCodeableConcept"].get("text", "")

            # Extract reference ranges if available
            if "referenceRange" in resource:
                ref_range = resource["referenceRange"][0]
                low = ref_range.get("low", {}).get("value", "")
                high = ref_range.get("high", {}).get("value", "")
                obs["reference_range"] = f"{low}-{high}" if low and high else ""

            observations.append(obs)

    return observations


def extract_conditions(bundle: dict) -> list:
    """Extract condition/diagnosis data from a FHIR Bundle."""
    conditions = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            condition = {
                "name": "",
                "code": "",
                "system": "",
                "clinical_status": "",
                "verification_status": "",
                "category": [],
                "onset_date": "",
                "abatement_date": "",
                "recorded_date": resource.get("recordedDate", ""),
            }

            # Extract condition code and name
            code_obj = resource.get("code", {})
            condition["name"] = code_obj.get("text", "")
            if "coding" in code_obj and code_obj["coding"]:
                coding = code_obj["coding"][0]
                condition["code"] = coding.get("code", "")
                condition["system"] = coding.get("system", "")
                if not condition["name"]:
                    condition["name"] = coding.get("display", "")

            # Extract clinical status
            clinical_status = resource.get("clinicalStatus", {})
            if "coding" in clinical_status and clinical_status["coding"]:
                condition["clinical_status"] = clinical_status["coding"][0].get("code", "")

            # Extract verification status
            verification = resource.get("verificationStatus", {})
            if "coding" in verification and verification["coding"]:
                condition["verification_status"] = verification["coding"][0].get("code", "")

            # Extract categories (primary, secondary, problem-list, etc.)
            categories = resource.get("category", [])
            for cat in categories:
                if "coding" in cat:
                    for coding in cat["coding"]:
                        condition["category"].append(coding.get("code", ""))

            # Extract onset date
            if "onsetDateTime" in resource:
                condition["onset_date"] = resource["onsetDateTime"]
            elif "onsetPeriod" in resource:
                condition["onset_date"] = resource["onsetPeriod"].get("start", "")

            # Extract abatement (resolution) date
            if "abatementDateTime" in resource:
                condition["abatement_date"] = resource["abatementDateTime"]

            conditions.append(condition)

    return conditions


def extract_medications(bundle: dict) -> list:
    """Extract medication data from a FHIR Bundle."""
    medications = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "MedicationRequest":
            med = {
                "medication": resource.get("medicationCodeableConcept", {}).get("text", "Unknown"),
                "status": resource.get("status", ""),
                "authoredOn": resource.get("authoredOn", ""),
                "dosageInstruction": "",
            }

            # Extract dosage
            if "dosageInstruction" in resource and resource["dosageInstruction"]:
                dosage = resource["dosageInstruction"][0]
                med["dosageInstruction"] = dosage.get("text", "")

            medications.append(med)

    return medications
