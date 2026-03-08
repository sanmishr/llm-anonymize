---
name: update-pipeline
description: This skill should be used when modifying anonymize.py, adding providers, fixing anonymization bugs, or syncing changes between the open-source repo and the toolkit copy.
version: 1.0.0
---

# Update Anonymization Pipeline

`anonymize.py` exists in TWO locations that must stay in sync:
1. **This repo** — open-source standalone (`/home/santz/Documents/llm-anonymize/anonymize.py`)
2. **Toolkit copy** — used by all projects (`/home/santz/Documents/local-ai-toolkit/tools/anonymize.py`)

## When to Use

- Any change to `anonymize.py` in either location
- Adding a new cloud LLM provider
- Fixing anonymization/de-anonymization bugs
- Updating entity detection patterns

## Sync Workflow

1. Make the change in whichever repo triggered it
2. **Copy to the other repo**: `cp anonymize.py <other-location>/anonymize.py`
3. Run tests in both locations
4. Commit both repos with matching commit messages

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
