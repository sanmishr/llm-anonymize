#!/usr/bin/env python3
"""Basic anonymize + deanonymize example."""

import sys
import os
import json

# Add parent directory so we can import anonymize.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anonymize import anonymize, deanonymize

# Sample text with sensitive entities
text = """
Hi team,

Following up on yesterday's call with Acme Corp. John Smith (john.smith@acme.com)
confirmed they want to proceed with Project Phoenix. Their CTO, Sarah Chen, wants
the Platform Engineering team to review the architecture by Friday.

The budget is $2.5M and they need delivery to their Austin office by Q3.
Contact: +1-555-0123.
"""

print("=== Original Text ===")
print(text)

# Anonymize
result = anonymize(text.strip())

print("=== Anonymized Text ===")
print(result["anonymized_text"])
print(f"\n--- {result['entity_count']} entities replaced ---\n")

print("=== Entity Mapping ===")
print(json.dumps(result["mapping"], indent=2))

# De-anonymize (simulating a cloud LLM response that uses the placeholders)
cloud_response = f"""Summary: {result['anonymized_text'][:100]}...

Action items:
1. Schedule architecture review with [PERSON_2]'s team
2. Confirm [AMOUNT_1] budget allocation
3. Ship to [LOCATION_1] by Q3
"""

print("\n=== Simulated Cloud Response (anonymized) ===")
print(cloud_response)

restored = deanonymize(cloud_response, result["mapping"])
print("=== De-anonymized Response ===")
print(restored)
