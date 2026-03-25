---
name: gdpr-compliance
description: Use when working with GDPR/EU AI Act compliance, audit logging, data subject rights (DSAR), retention policies, or the compliance.py module. Also triggered by "GDPR", "DPIA", "ROPA", "right to erasure", "data portability", "Art. 15/17/20/30".
version: 1.0.0
---

# GDPR & EU AI Act Compliance

`compliance.py` is the regulatory audit layer for the anonymization pipeline. It logs every `pipeline()` call and provides data subject rights tools.

## Architecture

```
anonymize.py pipeline()
    │
    ├── anonymize → call_cloud_llm → deanonymize (core flow)
    │
    └── compliance.log_pipeline_call()  ← only if ANONYMIZE_COMPLIANCE=1
            │
            └── anon_compliance_log table (SQLite default / PostgreSQL optional)
```

Two copies exist and must stay in sync:
1. **This repo** — standalone (`/home/santz/Documents/llm-anonymize/compliance.py`) — SQLite, zero deps
2. **Toolkit copy** — (`/home/santz/Documents/local-ai-toolkit/tools/anonymize_compliance.py`) — PostgreSQL via `pa_common`

## Key Files

| File | Purpose |
|------|---------|
| `compliance.py` | Standalone module: logging, DSAR, purge, reporting, CLI |
| `anonymize.py` | Core pipeline — calls `compliance.log_pipeline_call()` when enabled |
| `.env.example` | Config vars: `ANONYMIZE_COMPLIANCE`, `COMPLIANCE_DB`, `DATABASE_URL` |

Toolkit-specific (not in this repo):
| File | Purpose |
|------|---------|
| `tools/anonymize_compliance.py` | PostgreSQL version using `pa_common.get_conn()` |
| `tools/anonymize_migrate.py` | DB migration runner |
| `services/postgresql/init/12-create-compliance-tables.sql` | PostgreSQL schema |
| `harness/checks/compliance_audit.py` | Enforces `processing_purpose` on pipeline callers |

## GDPR Coverage

| Article | Feature | Function |
|---------|---------|----------|
| Art. 5 | Processing principles | `processing_purpose` + `lawful_basis` params |
| Art. 6 | Lawful basis | `lawful_basis` field (default: `legitimate_interest`) |
| Art. 15 | Right of access | `query_subject(identifier)` |
| Art. 17 | Right to erasure | `purge_subject(identifier)` |
| Art. 20 | Data portability | `export_subject(identifier)` → JSON |
| Art. 30 | Processing records | `log_processing()`, `summary()`, `ropa_report()` |
| Art. 35 | DPIA | Template at `docs/compliance/dpia-anonymization.md` (toolkit) |

## EU AI Act Coverage

| Article | Feature | Implementation |
|---------|---------|---------------|
| Art. 52 | Transparency | `ai_model_used` + `ai_model_version` (Ollama digest) in every record |
| Annex III | Risk classification | `risk_level` field: `minimal`, `low`, `high` |
| Art. 49 | Record-keeping | AI system card at `docs/compliance/ai-system-card.md` (toolkit) |

## Storage

- **Default**: SQLite at `~/.llm-anonymize/compliance.db` (auto-created)
- **PostgreSQL**: Set `DATABASE_URL=postgresql://user:pass@host/db`
- **No raw text stored** — only SHA-256 hash (`input_hash`) for correlation

## CLI Quick Reference

```bash
# Enable logging
export ANONYMIZE_COMPLIANCE=1

# Setup (auto on first use, but can run explicitly)
python compliance.py setup

# DSAR
python compliance.py query-subject john@acme.com       # Art. 15
python compliance.py purge-subject john@acme.com       # Art. 17
python compliance.py export-subject john@acme.com      # Art. 20

# Reporting
python compliance.py summary --days 30                 # Quick stats
python compliance.py ropa --days 365                   # Full ROPA report

# Retention
python compliance.py purge-expired --dry-run           # Preview
python compliance.py purge-expired                     # Delete expired
```

## Python API Quick Reference

```python
from compliance import (
    enable, is_enabled,          # Toggle logging
    log_pipeline_call,           # Auto-called by pipeline() when enabled
    log_processing,              # Manual logging (advanced)
    query_subject,               # Art. 15
    purge_subject,               # Art. 17
    export_subject,              # Art. 20
    summary, ropa_report,        # Art. 30
    purge_expired,               # Retention cleanup
    get_model_version,           # EU AI Act transparency
)
```

## Pipeline Integration

```python
from anonymize import pipeline

# Compliance params are optional — all have safe defaults
result = pipeline(
    "Review Acme's architecture with John",
    provider="deepseek",
    processing_purpose="architecture_review",
    lawful_basis="legitimate_interest",
    risk_level="low",
    subject_identifier="john@acme.com",
    retention_days=90,
)
```

## When Modifying

### Adding a new GDPR article
1. Add the field to `_SQLITE_SCHEMA` and `_PG_SCHEMA` in `compliance.py`
2. Add the parameter to `log_processing()` and `log_pipeline_call()`
3. Update `pipeline()` in `anonymize.py` to accept and pass it
4. Add CLI arg if user-facing
5. Sync to toolkit version

### Changing retention defaults
- Env var: `COMPLIANCE_RETENTION` (default: 90 days)
- Per-call: `retention_days` parameter in `pipeline()` or `log_processing()`
- Toolkit cron: `pa_cron.py compliance-purge` runs daily at 04:00

### Sync with toolkit
The open-source version (this repo) uses SQLite + self-contained DB logic.
The toolkit version uses `pa_common.get_conn()` + PostgreSQL + harness checks.
When changing compliance logic, update BOTH:
1. `compliance.py` (this repo — SQLite)
2. `tools/anonymize_compliance.py` (toolkit — PostgreSQL)

## Failures & Lessons

- **Compliance must never break the pipeline**: All logging is wrapped in try/except. A DB failure warns but does not prevent anonymization from working.
- **SQLite concurrent writes**: SQLite handles one writer at a time. For high-throughput use, switch to PostgreSQL via `DATABASE_URL`.
- **Purged != deleted**: `purge_subject()` sets `status='purged'` and NULLs PII fields but keeps the row for audit trail. The row itself is never physically deleted.
- **Model version caching**: `get_model_version()` calls Ollama `/api/show` and caches for 1 hour. If Ollama is down, returns `"unknown"` — this is intentional.
