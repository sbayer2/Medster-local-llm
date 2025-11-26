"""
Test script for Medster-local-LLM agent.
Tests the full multi-agent loop with a clinical reasoning query.
"""
import sys
sys.path.insert(0, '/Users/sbm4_mac/Desktop/Medster-local-LLM/src')

from medster.agent import Agent

# Test query that focuses on clinical reasoning (doesn't require database)
query = """
Analyze this clinical case:

58-year-old male with hypertension presents with acute chest pain.
- Pain: substernal, crushing, radiating to left arm
- Onset: 2 hours ago, at rest
- Associated symptoms: diaphoresis, nausea, dyspnea
- Vital signs: BP 160/95, HR 102, RR 22, O2 sat 94% on RA
- Risk factors: smoking 30 pack-years, family history of MI

What is the most likely diagnosis and what are the immediate next steps?
"""

print("=" * 70)
print("TESTING MEDSTER-LOCAL-LLM AGENT")
print("Model: gpt-oss:20b (local)")
print("=" * 70)
print()

# Create agent with conservative limits for testing
agent = Agent(max_steps=10, max_steps_per_task=3)

# Run the agent
print("Running clinical analysis...")
print()
result = agent.run(query)

print()
print("=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)
print()
print(result)
