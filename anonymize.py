#!/usr/bin/env python3
"""
llm-anonymize — Privacy layer for cloud LLM calls.

Uses a local LLM (via Ollama) to identify and replace sensitive entities with
placeholders like [COMPANY_1], [PERSON_1], etc. before text is sent to any cloud
LLM. Cloud responses are automatically de-anonymized back to the original names.

Works with any Ollama model that can follow instructions (default: qwen3:32b).
Supports any OpenAI-compatible cloud provider (DeepSeek, Groq, Gemini, MiniMax, etc.).

Quick start:
    # Anonymize text
    echo "Meeting with Acme Corp's John Smith at john@acme.com" | python anonymize.py anonymize

    # Full pipeline: anonymize → cloud LLM → de-anonymize
    echo "Review Acme's architecture" | python anonymize.py pipeline --provider deepseek --system "Summarize."

Python API:
    from anonymize import anonymize, deanonymize, pipeline
    result = anonymize("Meeting with Acme Corp's John Smith")
    print(result["anonymized_text"])  # "Meeting with [COMPANY_1]'s [PERSON_1]"

Configuration (env vars or .env file in current directory):
    ANONYMIZE_MODEL   — Ollama model (default: qwen3:32b)
    OLLAMA_URL        — Ollama endpoint (default: http://localhost:11434/api/generate)
    DEEPSEEK_API_KEY  — DeepSeek API key
    GROQ_API_KEY      — Groq API key
    GEMINI_API_KEY    — Gemini API key
    MINIMAX_API_KEY   — MiniMax API key

Full docs: https://github.com/santzstudios/llm-anonymize
"""

__version__ = "1.0.0"

import json
import os
import re
import sys
import logging
import argparse
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via env vars or .env file in current directory
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Load .env file from current directory if it exists. Simple key=value parser."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        # Don't override existing env vars
        if key not in os.environ:
            os.environ[key] = val

_load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("ANONYMIZE_MODEL", "qwen3:32b")

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_var": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_var": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "env_var": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M2.5",
    },
}

# ---------------------------------------------------------------------------
# Anonymization prompt
# ---------------------------------------------------------------------------

ANONYMIZE_PROMPT = """You are a privacy-protection assistant. Your ONLY job is to find and replace sensitive entities in the text below.

RULES:
1. Replace these entity types with numbered placeholders:
   - Company names → [COMPANY_1], [COMPANY_2], ...
   - Person names → [PERSON_1], [PERSON_2], ...
   - Email addresses → [EMAIL_1], [EMAIL_2], ...
   - Project/product names (internal) → [PROJECT_1], [PROJECT_2], ...
   - Team/department names → [TEAM_1], [TEAM_2], ...
   - Internal tool names → [TOOL_1], [TOOL_2], ...
   - Locations (offices, addresses) → [LOCATION_1], [LOCATION_2], ...
   - Phone numbers → [PHONE_1], [PHONE_2], ...
   - Account/ID numbers → [ACCOUNT_1], [ACCOUNT_2], ...
   - Monetary amounts with business context → [AMOUNT_1], [AMOUNT_2], ...

2. Keep these as-is (do NOT anonymize):
   - Public technology names: React, AWS, PostgreSQL, Docker, Kubernetes, etc.
   - Public standards: ISO, RFC, HTTP, REST, GraphQL, etc.
   - Generic roles: "architect", "developer", "manager" (without names)

3. Anonymize company-specific instances: "[COMPANY_1]'s AEM instance" not "Acme's AEM instance"

4. Be consistent: the same entity always gets the same placeholder.

5. When in doubt, ANONYMIZE. False positives are safer than data leaks.

Return ONLY valid JSON in this exact format, no other text:
{
  "anonymized_text": "the full text with all entities replaced by placeholders",
  "mapping": {"[COMPANY_1]": "Acme Corp", "[PERSON_1]": "John Smith"}
}

TEXT TO ANONYMIZE:
"""

# ---------------------------------------------------------------------------
# Regex safety net
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s.\-]{7,}\d\b")


def _regex_postpass(text: str, mapping: dict) -> tuple[str, dict]:
    """Safety net: catch emails and phone numbers the LLM may have missed."""
    def _next_placeholder(prefix: str, current_mapping: dict) -> str:
        nums = [
            int(k.strip("[]").split("_")[-1])
            for k in current_mapping
            if k.startswith(f"[{prefix}_")
        ]
        n = max(nums) + 1 if nums else 1
        return f"[{prefix}_{n}]"

    for match in _EMAIL_RE.findall(text):
        if match not in mapping.values():
            placeholder = _next_placeholder("EMAIL", mapping)
            mapping[placeholder] = match
            text = text.replace(match, placeholder)

    for match in _PHONE_RE.findall(text):
        cleaned = match.strip()
        if cleaned not in mapping.values():
            placeholder = _next_placeholder("PHONE", mapping)
            mapping[placeholder] = cleaned
            text = text.replace(match, placeholder)

    return text, mapping


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def anonymize(text: str, extra_entities: list[str] | None = None) -> dict:
    """
    Anonymize text using a local Ollama model.

    Args:
        text: The text to anonymize.
        extra_entities: Optional list of terms to force-replace even if the LLM misses them.

    Returns:
        dict with keys: anonymized_text, mapping, entity_count
    """
    prompt = ANONYMIZE_PROMPT + text

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()["response"]

    # Extract JSON from response
    try:
        start = result.index("{")
        end = result.rindex("}") + 1
        parsed = json.loads(result[start:end])
    except (ValueError, json.JSONDecodeError):
        logger.error(
            "Failed to parse LLM JSON response. "
            "Check that Ollama is running and the model (%s) is loaded.",
            MODEL,
        )
        return {"anonymized_text": text, "mapping": {}, "entity_count": 0}

    anon_text = parsed.get("anonymized_text", text)
    mapping = parsed.get("mapping", {})

    # Force-replace extra entities the user specified
    if extra_entities:
        def _next_num(prefix: str) -> int:
            nums = [
                int(k.strip("[]").split("_")[-1])
                for k in mapping
                if k.startswith(f"[{prefix}_")
            ]
            return max(nums) + 1 if nums else 1

        for entity in extra_entities:
            if entity in mapping.values():
                continue
            if entity not in anon_text:
                continue
            placeholder = f"[ENTITY_{_next_num('ENTITY')}]"
            mapping[placeholder] = entity
            anon_text = anon_text.replace(entity, placeholder)

    # Regex safety net
    anon_text, mapping = _regex_postpass(anon_text, mapping)

    return {
        "anonymized_text": anon_text,
        "mapping": mapping,
        "entity_count": len(mapping),
    }


def deanonymize(text: str, mapping: dict) -> str:
    """
    Replace placeholders back to original values.

    Args:
        text: Text containing [PLACEHOLDER] tokens.
        mapping: Dict mapping placeholders to original values.

    Returns:
        Text with all placeholders restored to originals.
    """
    sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)

    for placeholder in sorted_placeholders:
        original = mapping[placeholder]
        text = text.replace(f"{placeholder}'s", f"{original}'s")
        text = text.replace(placeholder, original)

    remaining = re.findall(r"\[[A-Z]+_\d+\]", text)
    if remaining:
        logger.warning("Unreplaced placeholders remain: %s", remaining)

    return text


def call_cloud_llm(
    prompt: str,
    provider: str,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """
    Send (already anonymized) prompt to a cloud LLM via OpenAI-compatible API.

    Args:
        prompt: The anonymized text to send.
        provider: One of: deepseek, groq, gemini, minimax.
        system_prompt: Optional system prompt.
        model: Override the provider's default model.
        max_tokens: Max response tokens (default: 2048).

    Returns:
        The cloud LLM's response text.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Options: {list(PROVIDERS.keys())}"
        )

    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_var"])
    if not api_key:
        sys.exit(
            f"ERROR: {cfg['env_var']} not set. "
            f"Export it or add to .env file."
        )
    model = model or cfg["default_model"]

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def pipeline(
    text: str,
    provider: str,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    extra_entities: list[str] | None = None,
) -> dict:
    """
    Full pipeline: anonymize → cloud LLM → de-anonymize.

    Args:
        text: Raw text with sensitive entities.
        provider: Cloud LLM provider (deepseek, groq, gemini, minimax).
        system_prompt: Optional system prompt for the cloud LLM.
        model: Override provider's default model.
        max_tokens: Max response tokens.
        extra_entities: Terms to force-anonymize.

    Returns:
        dict with keys: response, anonymized_text, mapping, raw_cloud_response
    """
    anon = anonymize(text, extra_entities=extra_entities)

    raw_response = call_cloud_llm(
        anon["anonymized_text"],
        provider=provider,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
    )

    final_response = deanonymize(raw_response, anon["mapping"])

    return {
        "response": final_response,
        "anonymized_text": anon["anonymized_text"],
        "mapping": anon["mapping"],
        "raw_cloud_response": raw_response,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_input(file_arg: str | None) -> str:
    """Read input from --file argument or stdin."""
    if file_arg:
        path = Path(file_arg)
        if not path.exists():
            sys.exit(f"ERROR: File not found: {file_arg}")
        return path.read_text().strip()
    elif not sys.stdin.isatty():
        return sys.stdin.read().strip()
    else:
        sys.exit("ERROR: Provide input via --file or stdin.")


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize text before sending to cloud LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- anonymize ---
    p_anon = subparsers.add_parser("anonymize", help="Anonymize text")
    p_anon.add_argument("--file", help="Input file (default: stdin)")
    p_anon.add_argument("--json", action="store_true", dest="json_output", help="Full JSON output")
    p_anon.add_argument("--entities", nargs="+", metavar="TERM", help="Force-catch terms")

    # --- deanonymize ---
    p_deanon = subparsers.add_parser("deanonymize", help="Restore placeholders")
    p_deanon.add_argument("--mapping", required=True, help="JSON mapping file")

    # --- pipeline ---
    p_pipe = subparsers.add_parser("pipeline", help="Anonymize + cloud LLM + de-anonymize")
    p_pipe.add_argument("--provider", required=True, choices=list(PROVIDERS.keys()))
    p_pipe.add_argument("--system", help="System prompt for cloud LLM")
    p_pipe.add_argument("--model", help="Override default model")
    p_pipe.add_argument("--max-tokens", type=int, default=2048, help="Max tokens (default: 2048)")
    p_pipe.add_argument("--entities", nargs="+", metavar="TERM", help="Force-catch terms")
    p_pipe.add_argument("--file", help="Input file (default: stdin)")
    p_pipe.add_argument("--json", action="store_true", dest="json_output", help="Full JSON output")

    args = parser.parse_args()

    if args.command == "anonymize":
        text = _read_input(args.file)
        result = anonymize(text, extra_entities=args.entities)

        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result["anonymized_text"])
            print(
                f"\n--- {result['entity_count']} entities replaced ---",
                file=sys.stderr,
            )

    elif args.command == "deanonymize":
        mapping_path = Path(args.mapping)
        if not mapping_path.exists():
            sys.exit(f"ERROR: Mapping file not found: {args.mapping}")
        mapping = json.loads(mapping_path.read_text())

        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            sys.exit("ERROR: Pipe anonymized text via stdin.")

        print(deanonymize(text, mapping))

    elif args.command == "pipeline":
        text = _read_input(args.file)
        result = pipeline(
            text,
            provider=args.provider,
            system_prompt=args.system,
            model=args.model,
            max_tokens=args.max_tokens,
            extra_entities=args.entities,
        )

        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result["response"])


if __name__ == "__main__":
    main()
