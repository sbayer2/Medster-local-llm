#!/usr/bin/env python3
"""Test script to verify Coherent Data Set path and data loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medster import config
from medster.tools.medical.api import COHERENT_DATA_PATH, list_available_patients

print("=" * 80)
print("COHERENT DATA PATH TEST")
print("=" * 80)

print(f"\n1. Config module path:")
print(f"   COHERENT_FHIR_PATH_ABS: {config.COHERENT_FHIR_PATH_ABS}")
print(f"   Path exists: {config.COHERENT_FHIR_PATH_ABS.exists()}")

print(f"\n2. API module path:")
print(f"   COHERENT_DATA_PATH: {COHERENT_DATA_PATH}")
print(f"   Path exists: {Path(COHERENT_DATA_PATH).exists()}")

print(f"\n3. Attempting to list patients...")
try:
    patients = list_available_patients(limit=5)
    print(f"   ✅ Found {len(patients)} patients (showing first 5)")
    for i, patient_id in enumerate(patients, 1):
        print(f"      {i}. {patient_id}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
