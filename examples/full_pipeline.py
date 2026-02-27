#!/usr/bin/env python3
"""
Full pipeline example: anonymize → cloud LLM → de-anonymize.

Requires:
  - Ollama running locally with a model (default: qwen3:32b)
  - A cloud LLM API key set (e.g., DEEPSEEK_API_KEY)

Usage:
  export DEEPSEEK_API_KEY="sk-..."
  python examples/full_pipeline.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anonymize import pipeline

text = """
Architecture review notes for Globex Corporation:

Their lead architect, Maria Garcia (maria@globex.com), presented the current system.
Key concerns:
- The "Atlas" microservice has a shared PostgreSQL database with "Hermes" — this
  creates tight coupling and makes independent deployment impossible.
- No observability stack. Their ops team (led by James Liu) is debugging production
  issues via SSH and manual log grepping.
- Authentication is handled by a custom LDAP bridge that hasn't been updated since 2019.

Recommendation: migrate to event-driven architecture with proper service boundaries.
Budget estimate: $800K over 6 months. Maria wants a formal proposal by March 15.
"""

print("=== Running full pipeline ===")
print(f"Input length: {len(text)} chars\n")

# Pick a provider (change to "groq", "gemini", or "minimax" as needed)
provider = "deepseek"

try:
    result = pipeline(
        text.strip(),
        provider=provider,
        system_prompt="You are a senior solutions architect. Summarize the key risks and provide 3 prioritized recommendations.",
        max_tokens=1024,
    )
except SystemExit as e:
    print(f"\n{e}")
    print(f"\nSet the API key and try again:")
    print(f"  export DEEPSEEK_API_KEY='sk-...'")
    sys.exit(1)

print("=== What the cloud LLM saw (anonymized) ===")
print(result["anonymized_text"][:500])
print("...\n")

print(f"=== Entity mapping ({len(result['mapping'])} entities) ===")
print(json.dumps(result["mapping"], indent=2))

print("\n=== Cloud LLM response (raw, still anonymized) ===")
print(result["raw_cloud_response"][:500])
print("...\n")

print("=== Final response (de-anonymized) ===")
print(result["response"])
