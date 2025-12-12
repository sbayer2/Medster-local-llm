#!/usr/bin/env python3
"""
Test script to verify allergy extraction using code generation primitives.
This demonstrates the pattern the agent should use when asked for allergies.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from medster.tools.analysis.primitives import load_patient, search_resources

def extract_allergies(patient_id: str):
    """Extract allergies for a patient using FHIR primitives."""

    # Load patient bundle
    bundle = load_patient(patient_id)

    # Extract AllergyIntolerance resources
    allergies = search_resources(bundle, 'AllergyIntolerance')

    # Format allergy data
    formatted_allergies = []
    for allergy in allergies:
        allergy_data = {
            'allergen': allergy.get('code', {}).get('text', 'Unknown'),
            'criticality': allergy.get('criticality', 'Unknown'),
            'category': allergy.get('category', ['Unknown'])[0] if allergy.get('category') else 'Unknown',
            'type': allergy.get('type', 'Unknown'),
            'onset': allergy.get('onsetDateTime', 'Unknown'),
        }

        # Extract reaction details
        reactions = allergy.get('reaction', [])
        if reactions:
            reaction = reactions[0]
            manifestations = reaction.get('manifestation', [])
            if manifestations:
                allergy_data['reaction'] = manifestations[0].get('text', 'Unknown')
            else:
                allergy_data['reaction'] = 'No specific reaction documented'
        else:
            allergy_data['reaction'] = 'No reaction documented'

        formatted_allergies.append(allergy_data)

    return {
        'patient_id': patient_id,
        'allergy_count': len(formatted_allergies),
        'allergies': formatted_allergies
    }


if __name__ == '__main__':
    # Test with the patient ID from the user's query
    patient_id = '39533e4a-f6f2-a144-ab37-6500460250dc'

    print(f"Extracting allergies for patient: {patient_id}\n")

    result = extract_allergies(patient_id)

    print(f"Found {result['allergy_count']} allergies:\n")

    for i, allergy in enumerate(result['allergies'], 1):
        print(f"{i}. {allergy['allergen']}")
        print(f"   Category: {allergy['category']}")
        print(f"   Criticality: {allergy['criticality']}")
        print(f"   Type: {allergy['type']}")
        print(f"   Reaction: {allergy['reaction']}")
        print(f"   Onset: {allergy['onset']}")
        print()
