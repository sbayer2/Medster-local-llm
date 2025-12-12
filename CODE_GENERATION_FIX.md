# Code Generation Fix - Summary

## Problem Identified

The agent was **recognizing tool limitations** but **not selecting `generate_and_run_analysis`** as the solution.

### Evidence from User's Test Run:

**Task 1 - AND Logic:**
```
Task: "Find patients with hypertension AND diabetes"
Agent thought: "analyze_batch_conditions can't do AND logic"
Agent action: ✗ No tool calls - marked complete
Expected: ✓ Should call generate_and_run_analysis
```

**Task 3 - Allergies:**
```
Task: "Fetch patient allergies"
Agent thought: "get_patient_conditions does not support allergy-specific filtering"
Agent action: ✗ No tool calls - marked complete
Expected: ✓ Should call generate_and_run_analysis
```

---

## Root Cause

**PROMPT ISSUE** - Not a logic issue

The prompts mentioned code generation but didn't provide:
1. Clear decision criteria for WHEN to use it
2. Concrete examples matching common scenarios
3. Guidance in task planning to avoid suggesting non-existent tools

---

## Changes Made

### 1. ACTION_SYSTEM_PROMPT - Added Decision Tree ✅

```
**DECISION TREE - When to Use generate_and_run_analysis:**

Ask yourself these questions IN ORDER:
1. ❓ "Is there a dedicated tool for this data type?"
   - Allergies, procedures, immunizations → ❌ NO TOOL EXISTS
   - **ACTION**: Use generate_and_run_analysis with search_resources()

2. ❓ "Does the task require AND/OR logic?"
   - "Patients with hypertension AND diabetes" → ❌ analyze_batch_conditions can't do AND
   - **ACTION**: Use generate_and_run_analysis with conditional filtering

3. ❓ "Does the task need cross-referencing multiple data sources?"
   - "Patients with diagnosis X AND have imaging/labs/ECG" → ❌ No tool does cross-referencing
   - **ACTION**: Use generate_and_run_analysis

4. ❓ "Does the available tool support the specific filter/parameter needed?"
   - get_patient_conditions doesn't filter allergies → ❌ Tool limitation
   - **ACTION**: Use generate_and_run_analysis

5. ❓ "Does the task involve vision/imaging analysis?"
   - Brain MRI analysis, ECG rhythm detection → ❌ Need to load images
   - **ACTION**: Use generate_and_run_analysis with vision primitives

**IF ANY ANSWER IS NO/LIMITATION FOUND → IMMEDIATELY call generate_and_run_analysis**
```

### 2. ACTION_SYSTEM_PROMPT - Added Concrete Examples ✅

```python
Example 1 - Allergies (no dedicated tool):
Task: "Fetch the patient's allergies"
Thought: "There is no get_patient_allergies tool"
Action: generate_and_run_analysis with code:
def analyze():
    bundle = load_patient(patient_id)
    allergies = search_resources(bundle, 'AllergyIntolerance')
    return {'allergies': allergies}

Example 2 - AND logic (tool limitation):
Task: "Find patients with hypertension AND diabetes"
Thought: "analyze_batch_conditions only searches one condition at a time"
Action: generate_and_run_analysis with code:
def analyze():
    patients = get_patients(patient_limit)
    matched = []
    for pid in patients:
        bundle = load_patient(pid)
        conditions = get_conditions(bundle)
        has_htn = any('hypertension' in c.get('display', '').lower() for c in conditions)
        has_dm = any('diabetes' in c.get('display', '').lower() for c in conditions)
        if has_htn and has_dm:
            matched.append(pid)
    return {'patients_with_both': matched}
```

### 3. PLANNING_SYSTEM_PROMPT - Added Tool Limitations ✅

```
**CRITICAL - Know When Tools Don't Exist:**
- NO TOOLS for: allergies, procedures, immunizations, care plans, family history
- For these data types → Plan task to use generate_and_run_analysis
- Example BAD task: "Fetch patient allergies using get_patient_conditions with filter 'allergy'"
- Example GOOD task: "Extract patient allergies using generate_and_run_analysis with FHIR AllergyIntolerance resources"

**CRITICAL - Know When Tools Have Limitations:**
- analyze_batch_conditions: ONLY single condition search, NO AND/OR logic
- Example BAD task: "Use analyze_batch_conditions to find patients with hypertension AND diabetes"
- Example GOOD task: "Use generate_and_run_analysis to find patients with hypertension AND diabetes using conditional logic"
```

---

## Test Results

✅ **Prompt validation passed**
- Decision tree present in ACTION_SYSTEM_PROMPT
- Concrete examples added for allergies and AND logic
- Planning guidance updated with tool limitations

---

## Recommended Test Prompts

### Test 1 - Allergies (No Dedicated Tool)
```
For patient 39533e4a-f6f2-a144-ab37-6500460250dc, extract their allergy information including allergen names and reactions.
```

**Expected**:
- Task planned: "Extract allergies using generate_and_run_analysis"
- Tool called: `generate_and_run_analysis`
- Code generated with: `search_resources(bundle, 'AllergyIntolerance')`

### Test 2 - AND Logic (Tool Limitation)
```
Find one patient with hypertension AND diabetes, show their age and medications.
```

**Expected**:
- Task planned: "Find patient with hypertension AND diabetes using generate_and_run_analysis"
- Tool called: `generate_and_run_analysis`
- Code generated with: conditional filtering for both conditions

### Test 3 - Procedures (No Dedicated Tool)
```
List all procedures performed on patient 39533e4a-f6f2-a144-ab37-6500460250dc.
```

**Expected**:
- Task planned: "Extract procedures using generate_and_run_analysis"
- Tool called: `generate_and_run_analysis`
- Code generated with: `search_resources(bundle, 'Procedure')`

---

## Files Modified

- ✅ `src/medster/prompts.py` - Added decision tree and concrete examples
- ✅ `test_code_generation_trigger.py` - Validation script
- ✅ `CODE_GENERATION_FIX.md` - This document

---

## Summary

**Issue**: Prompt guidance was too vague - agent recognized limitations but didn't know to use code generation

**Fix**: Added explicit decision tree, concrete examples, and planning guidance

**Result**: Agent should now automatically use `generate_and_run_analysis` when:
- No dedicated tool exists (allergies, procedures, immunizations)
- Tool has limitations (AND logic, cross-referencing)
- Filter/parameter not supported

**Test it** with the prompts above using ministral-3:8b! 🚀
