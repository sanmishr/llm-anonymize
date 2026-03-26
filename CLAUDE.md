# CLAUDE.md — llm-anonymize

Open-source privacy layer for cloud LLM calls. Uses local Ollama to strip PII before text reaches any cloud API.

**Repo:** `github.com/sanmishr/llm-anonymize` | **License:** MIT | **Version:** 2.0.0

## Architecture

```
User text → Local LLM (anonymize) → Cloud LLM → Local LLM (de-anonymize) → Clean response
                                        │
                              compliance.log_pipeline_call()  ← if ANONYMIZE_COMPLIANCE=1
                                        │
                              anon_compliance_log (SQLite / PostgreSQL)
```

Two files: `anonymize.py` (core pipeline) + `compliance.py` (GDPR/EU AI Act audit layer).

## Key Rules

- **No personal data in examples or tests** — use synthetic data only.
- **Supports 4 providers:** DeepSeek, Groq, Gemini, MiniMax (any OpenAI-compatible).
- **Any Ollama model works** — not locked to Qwen3:32B.
- **Compliance is opt-in** — set `ANONYMIZE_COMPLIANCE=1` to enable audit logging.
- **Compliance never breaks the pipeline** — logging failures warn, never block.

## Usage

```bash
# CLI
echo "Meeting with John Smith at Acme Corp" | python anonymize.py pipeline --provider groq

# With GDPR compliance
ANONYMIZE_COMPLIANCE=1 echo "Review proposal" | python anonymize.py pipeline --provider deepseek --purpose review

# Python API
from anonymize import pipeline, anonymize, deanonymize, call_cloud_llm
from compliance import enable, summary, query_subject, purge_subject
```

## Known Failure Patterns

- **Ollama not running**: Check `curl http://localhost:11434/api/tags`. Start with `systemctl start ollama`.
- **Model not pulled**: `ollama pull qwen3:32b` (or any instruction-following model).
- **Incomplete anonymization**: Some entity types (project codenames, internal jargon) may slip through. The local LLM does best-effort extraction.
- **API key missing**: Keys loaded from env vars (`DEEPSEEK_API_KEY`, `GROQ_API_KEY`, etc). Check `.env` files.
- **SQLite concurrent writes**: For high-throughput, switch to PostgreSQL via `DATABASE_URL`.
