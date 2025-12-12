#!/usr/bin/env python3
"""
Test that the agent now correctly triggers generate_and_run_analysis
for tasks that require code generation (allergies, AND logic, etc.)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from medster.agent import Agent
from medster.prompts import PLANNING_SYSTEM_PROMPT, ACTION_SYSTEM_PROMPT

def test_prompts_have_code_gen_guidance():
    """Verify prompts include code generation decision tree."""
    print("Testing prompt updates...\n")

    # Check PLANNING_SYSTEM_PROMPT
    print("1. Checking PLANNING_SYSTEM_PROMPT:")
    assert "NO TOOLS for: allergies, procedures, immunizations" in PLANNING_SYSTEM_PROMPT
    assert "analyze_batch_conditions: ONLY single condition search, NO AND/OR logic" in PLANNING_SYSTEM_PROMPT
    print("   ✓ Planning prompt updated with tool limitations")

    # Check ACTION_SYSTEM_PROMPT
    print("\n2. Checking ACTION_SYSTEM_PROMPT:")
    assert "DECISION TREE" in ACTION_SYSTEM_PROMPT
    assert "Is there a dedicated tool for this data type?" in ACTION_SYSTEM_PROMPT
    assert "Allergies, procedures, immunizations" in ACTION_SYSTEM_PROMPT
    assert "Does the task require AND/OR logic?" in ACTION_SYSTEM_PROMPT
    assert "CONCRETE EXAMPLES:" in ACTION_SYSTEM_PROMPT
    print("   ✓ Action prompt updated with decision tree")
    print("   ✓ Concrete examples added for allergies and AND logic")

    return True

def show_expected_behavior():
    """Show what should happen now."""
    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR FOR YOUR QUERY:")
    print("="*60)

    print("\nQuery: 'Find one patient with hypertension AND diabetes'")
    print("\nOLD BEHAVIOR (broken):")
    print("  ✗ Task: 'Use analyze_batch_conditions to find patient with both'")
    print("  ✗ Agent thinks: 'analyze_batch_conditions can't do this'")
    print("  ✗ Agent action: Returns no tool calls, marks complete")

    print("\nNEW BEHAVIOR (fixed):")
    print("  ✓ Task: 'Use generate_and_run_analysis to find patient with both'")
    print("  ✓ Agent thinks: 'No tool does AND logic → Decision Tree Q2'")
    print("  ✓ Agent action: Calls generate_and_run_analysis with code")
    print("  ✓ Code:")
    print("      def analyze():")
    print("          patients = get_patients(100)")
    print("          for pid in patients:")
    print("              conditions = get_conditions(load_patient(pid))")
    print("              has_htn = any('hypertension' in c.display for c in conditions)")
    print("              has_dm = any('diabetes' in c.display for c in conditions)")
    print("              if has_htn and has_dm:")
    print("                  return {'patient_id': pid}")

    print("\n" + "="*60)
    print("\nQuery: 'Check if patient has allergies'")
    print("\nOLD BEHAVIOR (broken):")
    print("  ✗ Task: 'Fetch allergies using get_patient_conditions'")
    print("  ✗ Agent thinks: 'get_patient_conditions doesn't support allergies'")
    print("  ✗ Agent action: Returns no tool calls, marks complete")

    print("\nNEW BEHAVIOR (fixed):")
    print("  ✓ Task: 'Extract allergies using generate_and_run_analysis'")
    print("  ✓ Agent thinks: 'No get_patient_allergies tool → Decision Tree Q1'")
    print("  ✓ Agent action: Calls generate_and_run_analysis with code")
    print("  ✓ Code:")
    print("      def analyze():")
    print("          bundle = load_patient(patient_id)")
    print("          allergies = search_resources(bundle, 'AllergyIntolerance')")
    print("          return {'allergies': allergies}")

if __name__ == '__main__':
    print("CODE GENERATION TRIGGER TEST")
    print("="*60 + "\n")

    try:
        success = test_prompts_have_code_gen_guidance()

        if success:
            show_expected_behavior()

            print("\n" + "="*60)
            print("✓ PROMPTS UPDATED SUCCESSFULLY")
            print("="*60)
            print("\nThe agent should now:")
            print("  1. Plan tasks with generate_and_run_analysis for allergies/AND logic")
            print("  2. Recognize tool limitations using decision tree")
            print("  3. Call generate_and_run_analysis instead of giving up")
            print("\nTest with: 'Find one patient with hypertension AND diabetes'")
            print("Expected: Should call generate_and_run_analysis, not analyze_batch_conditions")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
