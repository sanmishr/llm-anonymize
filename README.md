# llm-anonymize

Privacy layer for cloud LLM calls. Uses a **local LLM** (via Ollama) to strip sensitive entities from your text before it reaches any cloud API. De-anonymizes the response automatically.

Your data never leaves your machine unprotected.

```
                        ┌──────────────┐
  "Meeting with         │  Local LLM   │    "[COMPANY_1]'s [PERSON_1]
   Acme Corp's    ───►  │  (Ollama)    │ ──► wants to discuss [PROJECT_1]"
   John Smith"          │  anonymize   │
                        └──────────────┘
                                              │
                                              ▼
                        ┌──────────────┐    ┌──────────────┐
  "Acme Corp's         │  Local LLM   │    │  Cloud LLM   │
   John Smith     ◄──  │  de-anonymize │ ◄──│  (DeepSeek,  │
   should..."           │              │    │  Groq, etc.) │
                        └──────────────┘    └──────────────┘
```

## Quick Start

```bash
# 1. Install Ollama and pull a model (any instruction-following model works)
ollama pull qwen3:32b

# 2. Install dependencies
pip install openai requests

# 3. Anonymize!
echo "Meeting with Acme Corp's John Smith at john@acme.com" | python anonymize.py anonymize
```

Output:
```
Meeting with [COMPANY_1]'s [PERSON_1] at [EMAIL_1]

--- 3 entities replaced ---
```

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** running locally with any instruction-following model
- **openai** Python package (for cloud LLM calls via the `pipeline` command)
- **requests** Python package

## Usage

### CLI — Anonymize

```bash
# From stdin
echo "Email john@acme.com about Project Phoenix" | python anonymize.py anonymize

# From file
python anonymize.py anonymize --file meeting-notes.txt

# JSON output (includes full mapping)
echo "Call Jane Doe at Globex" | python anonymize.py anonymize --json

# Force-catch specific terms the LLM might miss
echo "Discuss the Nightingale initiative" | python anonymize.py anonymize --entities "Nightingale"
```

### CLI — De-anonymize

```bash
# Save the mapping first
echo "Call Jane Doe at Globex" | python anonymize.py anonymize --json > result.json

# Then de-anonymize any text using that mapping
echo "[PERSON_1] confirmed the [COMPANY_1] deal" | python anonymize.py deanonymize --mapping mapping.json
```

### CLI — Full Pipeline

Anonymize → send to cloud LLM → de-anonymize the response, all in one command:

```bash
# Requires API key set (env var or .env file)
export DEEPSEEK_API_KEY="sk-..."

echo "Review Acme Corp's architecture proposal" | \
  python anonymize.py pipeline --provider deepseek --system "Summarize this."

# Other providers
echo "..." | python anonymize.py pipeline --provider groq
echo "..." | python anonymize.py pipeline --provider gemini
echo "..." | python anonymize.py pipeline --provider minimax

# Override model
echo "..." | python anonymize.py pipeline --provider groq --model llama-3.3-70b-versatile

# Full JSON output (includes anonymized text, mapping, raw cloud response)
echo "..." | python anonymize.py pipeline --provider deepseek --json
```

### Python API

```python
from anonymize import anonymize, deanonymize, call_cloud_llm, pipeline

# --- Anonymize ---
result = anonymize("Meeting with Acme Corp's John Smith at john@acme.com")
print(result["anonymized_text"])   # "Meeting with [COMPANY_1]'s [PERSON_1] at [EMAIL_1]"
print(result["mapping"])           # {"[COMPANY_1]": "Acme Corp", "[PERSON_1]": "John Smith", ...}
print(result["entity_count"])      # 3

# --- De-anonymize ---
restored = deanonymize("[PERSON_1] confirmed the deal", result["mapping"])
print(restored)  # "John Smith confirmed the deal"

# --- Call cloud LLM (with pre-anonymized text) ---
response = call_cloud_llm(
    result["anonymized_text"],
    provider="deepseek",
    system_prompt="Summarize this meeting note.",
)
final = deanonymize(response, result["mapping"])

# --- Full pipeline (one call) ---
result = pipeline(
    "Review Acme Corp's architecture with John Smith",
    provider="deepseek",
    system_prompt="Provide a technical summary.",
)
print(result["response"])            # De-anonymized response
print(result["raw_cloud_response"])  # What the cloud actually saw/returned
print(result["mapping"])             # The entity mapping used
```

## Supported Providers

Any OpenAI-compatible API works. These are pre-configured:

| Provider | Env Var | Default Model | Base URL |
|----------|---------|---------------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` | `https://api.deepseek.com/v1` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| MiniMax | `MINIMAX_API_KEY` | `MiniMax-M2.5` | `https://api.minimax.io/v1` |

### Adding a Custom Provider

Edit the `PROVIDERS` dict in `anonymize.py`:

```python
PROVIDERS["my-provider"] = {
    "base_url": "https://api.example.com/v1",
    "env_var": "MY_PROVIDER_API_KEY",
    "default_model": "my-model-name",
}
```

Any endpoint that speaks the OpenAI chat completions format will work.

## How It Works

### 1. LLM-Based Entity Detection

The local LLM receives a carefully crafted prompt that instructs it to identify and replace sensitive entities across 10 categories:

| Category | Placeholder | Example |
|----------|-------------|---------|
| Companies | `[COMPANY_1]` | Acme Corp → [COMPANY_1] |
| People | `[PERSON_1]` | John Smith → [PERSON_1] |
| Emails | `[EMAIL_1]` | john@acme.com → [EMAIL_1] |
| Projects | `[PROJECT_1]` | Project Phoenix → [PROJECT_1] |
| Teams | `[TEAM_1]` | Platform Engineering → [TEAM_1] |
| Tools | `[TOOL_1]` | Internal tools → [TOOL_1] |
| Locations | `[LOCATION_1]` | 123 Main St → [LOCATION_1] |
| Phone numbers | `[PHONE_1]` | +1-555-0123 → [PHONE_1] |
| Account IDs | `[ACCOUNT_1]` | ACC-12345 → [ACCOUNT_1] |
| Monetary amounts | `[AMOUNT_1]` | $2.5M deal → [AMOUNT_1] |

Public technology names (AWS, Docker, React, etc.) and generic roles are preserved.

### 2. Regex Safety Net

After the LLM pass, a regex post-processor catches any emails or phone numbers the LLM might have missed. Belt and suspenders.

### 3. Force-Catch Entities

The `--entities` flag lets you specify terms that MUST be anonymized, even if the LLM considers them public. Useful for internal project codenames that look like common words.

### 4. Consistent Mapping

The same entity always gets the same placeholder within a single anonymization call. The mapping is returned so you can de-anonymize any response that references those entities.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ANONYMIZE_MODEL` | `qwen3:32b` | Ollama model for entity detection |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GROQ_API_KEY` | — | Groq API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `MINIMAX_API_KEY` | — | MiniMax API key |

Set via environment variables or a `.env` file in the current directory.

## FAQ

**What Ollama models work?**

Any model that can follow instructions and output JSON. Tested with:
- `qwen3:32b` — best quality, recommended if you have the VRAM (~20GB)
- `qwen3:8b` — good quality, runs on most GPUs (~5GB)
- `llama3.1:8b` — works well, widely available
- `mistral:7b` — works, occasionally misses edge cases

Larger models catch more entities. Even 7B models handle the common cases (names, emails, companies).

**Can I use this without Ollama?**

The `anonymize()` function calls Ollama's REST API. If you have another local LLM server that speaks the same `/api/generate` endpoint format, point `OLLAMA_URL` at it. For any other setup, you'd need to modify the `anonymize()` function.

**How good is the anonymization?**

The LLM catches ~95% of entities on first pass. The regex safety net catches remaining emails and phone numbers. The `--entities` flag handles known terms. For maximum coverage, use a larger model and the `--entities` flag for domain-specific terms.

No anonymization is perfect — always review the output for high-stakes use cases.

**Can I add my own entity categories?**

Edit the `ANONYMIZE_PROMPT` in `anonymize.py`. Add your category to the numbered list (e.g., `- Medical record numbers → [MRN_1], [MRN_2], ...`). The LLM will pick it up.

**Is the mapping stored anywhere?**

Only in memory during execution. Use `--json` to capture it, or save `result["mapping"]` in your code. Nothing is written to disk unless you explicitly save it.

## License

MIT — see [LICENSE](LICENSE).
