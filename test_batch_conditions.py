#!/usr/bin/env python3
"""Test the analyze_batch_conditions tool directly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medster.tools.medical.patient_data import analyze_batch_conditions

print("=" * 80)
print("TESTING analyze_batch_conditions")
print("=" * 80)

print("\nTest 1: Search for 'hypertension,diabetes' (OR logic)")

# Call the tool's run method
result = analyze_batch_conditions.run({
    "patient_limit": 50,
    "condition_filter": "hypertension,diabetes"
})

print(f"\nPatients analyzed: {result.get('patients_analyzed', 0)}")
print(f"Total condition occurrences: {result.get('total_condition_occurrences', 0)}")
print(f"Unique conditions: {result.get('unique_conditions', 0)}")
print(f"Filter applied: {result.get('filter_applied', 'None')}")

if 'error' in result:
    print(f"\n❌ ERROR: {result['error']}")
    import traceback
    if 'traceback' in result:
        print(result['traceback'])
else:
    print(f"\n✅ Top conditions found:")
    for i, cond in enumerate(result.get('most_common_conditions', [])[:10], 1):
        print(f"   {i}. {cond['condition']}")
        print(f"      - Occurrences: {cond['occurrence_count']}")
        print(f"      - Patients: {cond['patient_count']}")
        print(f"      - Sample IDs: {cond['patient_ids'][:3]}")

print("\n" + "=" * 80)
