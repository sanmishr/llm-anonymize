# CLAUDE.md — llm-anonymize

Open-source privacy layer for cloud LLM calls. Uses local Ollama to strip PII before text reaches any cloud API.

**Repo:** `github.com/sanmishr/llm-anonymize` | **License:** MIT | **Version:** 1.0.0

## Architecture

```
User text → Local LLM (anonymize) → Cloud LLM → Local LLM (de-anonymize) → Clean response
```

Single file: `anonymize.py` (457 lines). CLI + importable Python API.

## Key Rules

- **This is the canonical open-source version** of the toolkit's `anonymize.py`. Changes here should be mirrored to `/home/santz/Documents/local-ai-toolkit/tools/anonymize.py` and vice versa.
- **No personal data in examples or tests** — use synthetic data only.
- **Supports 4 providers:** DeepSeek, Groq, Gemini, OpenAI-compatible.
- **Any Ollama model works** — not locked to Qwen3:32B.

## Usage

```bash
# CLI
echo "Meeting with John Smith at Acme Corp" | python anonymize.py pipeline --provider groq

# Python API
from anonymize import pipeline, anonymize, deanonymize, call_cloud_llm
```

## Known Failure Patterns

- **Ollama not running**: Check `curl http://localhost:11434/api/tags`. Start with `systemctl start ollama`.
- **Model not pulled**: `ollama pull qwen3:32b` (or any instruction-following model).
- **Incomplete anonymization**: Some entity types (project codenames, internal jargon) may slip through. The local LLM does best-effort extraction.
- **API key missing**: Keys loaded from env vars (`DEEPSEEK_API_KEY`, `GROQ_API_KEY`, etc). Check `.env` files.
