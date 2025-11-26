"""
Test the model selection feature.
This simulates selecting gpt-oss:20b and running a simple query.
"""
import sys
sys.path.insert(0, '/Users/sbm4_mac/Desktop/Medster-local-LLM/src')

from medster.agent import Agent

print("=" * 70)
print("TESTING MODEL SELECTION FEATURE")
print("=" * 70)
print()

# Test 1: Create agent with text-only model (gpt-oss:20b)
print("Test 1: Creating agent with gpt-oss:20b (text-only)")
agent_text = Agent(model_name="gpt-oss:20b")
print(f"✓ Agent created with model: {agent_text.model_name}")
print()

# Test 2: Create agent with vision model (qwen3-vl:8b)
print("Test 2: Creating agent with qwen3-vl:8b (text + vision)")
try:
    agent_vision = Agent(model_name="qwen3-vl:8b")
    print(f"✓ Agent created with model: {agent_vision.model_name}")
    print("  (Note: Vision model not pulled yet, but agent initialization works)")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 3: Run a simple text-only query with gpt-oss:20b
print("Test 3: Running simple query with gpt-oss:20b")
print("-" * 70)
query = "What are the main causes of chest pain in a 58-year-old male?"
print(f"Query: {query}")
print()
print("Running analysis...")

try:
    result = agent_text.run(query)
    print()
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print("Results:")
    print(result)
    print()
    print("=" * 70)
    print("Model selection feature is working correctly!")
    print("=" * 70)
except Exception as e:
    print(f"✗ Error during analysis: {e}")
    import traceback
    traceback.print_exc()
