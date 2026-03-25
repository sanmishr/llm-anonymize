---
name: update-pipeline
description: This skill should be used when modifying anonymize.py, adding providers, fixing anonymization bugs, or syncing changes between the open-source repo and the toolkit copy.
version: 1.0.0
---

# Update Anonymization Pipeline

## When to Use

- Any change to `anonymize.py`
- Adding a new cloud LLM provider
- Fixing anonymization/de-anonymization bugs
- Updating entity detection patterns

## Key Files

| File | Purpose |
|------|---------|
| `anonymize.py` | Core pipeline: anonymize → cloud LLM → de-anonymize |
| `compliance.py` | GDPR/EU AI Act audit logging (optional) |
| `.env.example` | API keys and config vars |

## After Changes

1. Run `python anonymize.py anonymize --json` with test input to verify entity detection
2. Run `python anonymize.py pipeline --provider <provider>` to verify end-to-end
3. If compliance logic changed, test with `ANONYMIZE_COMPLIANCE=1`

## Supported Providers

| Provider | Env Var | Base URL |
|----------|---------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| Groq | `GROQ_API_KEY` | `https://api.groq.com/openai` |
| Gemini | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com` |
| OpenAI-compatible | `OPENAI_API_KEY` | configurable |

## Failures & Lessons

- **Incomplete anonymization**: Local LLM does best-effort entity extraction. Unusual entity types (internal project codenames, acronyms) may slip through. Test with real-world examples.
- **De-anonymization mismatch**: If anonymized text is truncated or reformatted by the cloud LLM, placeholders like `[PERSON_1]` may not match on de-anonymize. The pipeline handles partial matches but edge cases exist.
- **Two-repo sync drift**: If you change one copy and forget the other, they diverge silently. Always sync both in the same session.
- **Ollama model matters**: Any instruction-following model works, but larger models (32B+) are more reliable at entity detection than 7B models.
