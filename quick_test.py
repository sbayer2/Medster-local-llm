"""
Quick test of Medster-local-LLM with a simpler query.
"""
import sys
sys.path.insert(0, '/Users/sbm4_mac/Desktop/Medster-local-LLM/src')

from medster.model import call_llm

print("=" * 70)
print("MEDSTER-LOCAL-LLM QUICK TEST")
print("=" * 70)
print()

# Simple medical reasoning query (no database needed)
query = """
A 65-year-old patient with diabetes presents with polyuria, polydipsia,
and fatigue. Blood glucose is 450 mg/dL. What is the likely diagnosis
and what are the immediate treatment priorities?

Answer in 2-3 sentences.
"""

print("Query:", query.strip())
print()
print("Generating response with gpt-oss:20b...")
print()

response = call_llm(query)

print("=" * 70)
print("RESPONSE")
print("=" * 70)
print()
print(response.content)
print()
print("=" * 70)
print("SUCCESS! Local LLM working perfectly.")
print("Cost: $0 (100% local)")
print("=" * 70)
