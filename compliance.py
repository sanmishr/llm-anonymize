#!/usr/bin/env python3
"""
llm-anonymize compliance — GDPR & EU AI Act audit layer.

Logs every anonymization pipeline call with structured metadata required by
GDPR Article 30 (Records of Processing Activities) and EU AI Act Article 52
(transparency). Supports data subject rights: access (Art. 15), erasure
(Art. 17), and portability (Art. 20).

Works with SQLite (zero config, default) or PostgreSQL (via DATABASE_URL).

Quick start:
    # Enable compliance logging (just set the env var)
    export ANONYMIZE_COMPLIANCE=1

    # Or use the Python API
    from compliance import enable, query_subject, purge_subject, summary

    enable()  # starts logging all pipeline() calls
    summary()  # view processing summary

CLI:
    python compliance.py setup                     # Create/verify DB schema
    python compliance.py summary [--days 30]       # Processing summary
    python compliance.py ropa [--days 365]         # Full ROPA report
    python compliance.py query-subject IDENTIFIER  # Art. 15: access request
    python compliance.py purge-subject IDENTIFIER  # Art. 17: erasure
    python compliance.py export-subject IDENTIFIER # Art. 20: portability
    python compliance.py purge-expired [--dry-run] # Retention cleanup

Configuration (env vars):
    ANONYMIZE_COMPLIANCE   — Set to "1" to enable logging (default: off)
    COMPLIANCE_DB          — SQLite path (default: ~/.llm-anonymize/compliance.db)
    DATABASE_URL           — PostgreSQL URL (overrides SQLite if set)
    COMPLIANCE_RETENTION   — Days to keep records (default: 90)

Full docs: https://github.com/sanmishr/llm-anonymize
"""

__version__ = "1.0.0"

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_DEFAULT_MODEL = os.environ.get("ANONYMIZE_MODEL", "qwen3:32b")
_DEFAULT_RETENTION = int(os.environ.get("COMPLIANCE_RETENTION", "90"))
_DEFAULT_DB_PATH = Path.home() / ".llm-anonymize" / "compliance.db"

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS anon_compliance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- GDPR Art. 30
    processing_purpose   TEXT NOT NULL,
    lawful_basis         TEXT NOT NULL DEFAULT 'legitimate_interest',
    consent_reference    TEXT,
    recipients           TEXT NOT NULL DEFAULT '[]',
    entity_categories    TEXT DEFAULT '[]',
    entity_count         INTEGER DEFAULT 0,
    retention_days       INTEGER DEFAULT 90,
    anonymization_method TEXT DEFAULT 'local_llm_ner_placeholder',
    expires_at           TEXT,

    -- EU AI Act
    ai_model_used        TEXT,
    ai_model_version     TEXT,
    risk_level           TEXT DEFAULT 'low',

    -- Operational
    session_id           TEXT,
    caller_module        TEXT,
    input_hash           TEXT,
    subject_identifier   TEXT,
    status               TEXT DEFAULT 'completed',
    detail               TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_acl_subject    ON anon_compliance_log(subject_identifier);
CREATE INDEX IF NOT EXISTS idx_acl_expires    ON anon_compliance_log(expires_at);
CREATE INDEX IF NOT EXISTS idx_acl_created    ON anon_compliance_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_acl_purpose    ON anon_compliance_log(processing_purpose);
"""

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS anon_compliance_log (
    id                   SERIAL PRIMARY KEY,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    processing_purpose   TEXT NOT NULL,
    lawful_basis         TEXT NOT NULL DEFAULT 'legitimate_interest',
    consent_reference    TEXT,
    recipients           TEXT[] NOT NULL DEFAULT '{}',
    entity_categories    TEXT[] DEFAULT '{}',
    entity_count         INTEGER DEFAULT 0,
    retention_days       INTEGER DEFAULT 90,
    anonymization_method TEXT DEFAULT 'local_llm_ner_placeholder',
    expires_at           TIMESTAMPTZ,
    ai_model_used        TEXT,
    ai_model_version     TEXT,
    risk_level           TEXT DEFAULT 'low',
    session_id           TEXT,
    caller_module        TEXT,
    input_hash           TEXT,
    subject_identifier   TEXT,
    status               TEXT DEFAULT 'completed',
    detail               JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_acl_subject ON anon_compliance_log(subject_identifier);
CREATE INDEX IF NOT EXISTS idx_acl_expires ON anon_compliance_log(expires_at);
CREATE INDEX IF NOT EXISTS idx_acl_created ON anon_compliance_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_acl_purpose ON anon_compliance_log(processing_purpose);
"""


# ---------------------------------------------------------------------------
# Database abstraction (SQLite default, PostgreSQL optional)
# ---------------------------------------------------------------------------

def _is_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _get_conn():
    """Get a database connection (SQLite or PostgreSQL)."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        import psycopg2
        return psycopg2.connect(db_url)

    db_path = Path(os.environ.get("COMPLIANCE_DB", str(_DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    """Create compliance table. Idempotent."""
    conn = _get_conn()
    cur = conn.cursor()
    schema = _PG_SCHEMA if _is_postgres() else _SQLITE_SCHEMA
    cur.executescript(schema) if not _is_postgres() else cur.execute(schema)
    conn.commit()
    conn.close()
    logger.info("Compliance DB schema ready.")


def _ensure_db():
    """Auto-create schema on first use."""
    if _is_postgres():
        setup_db()
        return
    db_path = Path(os.environ.get("COMPLIANCE_DB", str(_DEFAULT_DB_PATH)))
    if not db_path.exists():
        setup_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(retention_days: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(days=retention_days)).isoformat()


def _serialize_list(items: list) -> str:
    """Serialize list for DB storage (JSON for SQLite, native for PG)."""
    if _is_postgres():
        return items  # psycopg2 handles list → array
    return json.dumps(items)


# ---------------------------------------------------------------------------
# EU AI Act: Model Transparency
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cached_model_version(model: str, _hour_key: int) -> str:
    try:
        resp = requests.post(
            f"{_OLLAMA_URL}/api/show",
            json={"name": model},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("digest", "unknown")[:12]
    except Exception:
        return "unknown"


def get_model_version(model: str | None = None) -> str:
    """Get Ollama model digest (cached 1 hour). EU AI Act transparency."""
    model = model or _DEFAULT_MODEL
    hour_key = int(datetime.now(timezone.utc).timestamp()) // 3600
    return _cached_model_version(model, hour_key)


# ---------------------------------------------------------------------------
# Art. 30: Log Processing
# ---------------------------------------------------------------------------

def log_processing(
    *,
    processing_purpose: str,
    recipients: list[str],
    lawful_basis: str = "legitimate_interest",
    entity_categories: list[str] | None = None,
    entity_count: int = 0,
    retention_days: int | None = None,
    anonymization_method: str = "local_llm_ner_placeholder",
    ai_model_used: str | None = None,
    ai_model_version: str | None = None,
    risk_level: str = "low",
    session_id: str | None = None,
    caller_module: str | None = None,
    input_hash: str | None = None,
    subject_identifier: str | None = None,
    consent_reference: str | None = None,
    detail: dict | None = None,
) -> int | None:
    """Insert a GDPR Art. 30 compliant processing record. Returns row ID."""
    _ensure_db()
    retention = retention_days or _DEFAULT_RETENTION

    conn = _get_conn()
    cur = conn.cursor()
    try:
        if _is_postgres():
            cur.execute(
                """INSERT INTO anon_compliance_log
                   (processing_purpose, lawful_basis, consent_reference, recipients,
                    entity_categories, entity_count, retention_days, anonymization_method,
                    ai_model_used, ai_model_version, risk_level, expires_at,
                    session_id, caller_module, input_hash, subject_identifier, detail)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()+(%s||' days')::INTERVAL,
                           %s,%s,%s,%s,%s)
                   RETURNING id""",
                (processing_purpose, lawful_basis, consent_reference, recipients,
                 entity_categories or [], entity_count, retention, anonymization_method,
                 ai_model_used, ai_model_version, risk_level, str(retention),
                 session_id, caller_module, input_hash, subject_identifier,
                 json.dumps(detail or {})),
            )
            row_id = cur.fetchone()[0]
        else:
            cur.execute(
                """INSERT INTO anon_compliance_log
                   (processing_purpose, lawful_basis, consent_reference, recipients,
                    entity_categories, entity_count, retention_days, anonymization_method,
                    ai_model_used, ai_model_version, risk_level, expires_at,
                    session_id, caller_module, input_hash, subject_identifier, detail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (processing_purpose, lawful_basis, consent_reference,
                 _serialize_list(recipients), _serialize_list(entity_categories or []),
                 entity_count, retention, anonymization_method,
                 ai_model_used, ai_model_version, risk_level, _expires_iso(retention),
                 session_id, caller_module, input_hash, subject_identifier,
                 json.dumps(detail or {})),
            )
            row_id = cur.lastrowid
        conn.commit()
        return row_id
    except Exception:
        conn.rollback()
        logger.warning("Compliance logging failed.", exc_info=True)
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Art. 15: Right of Access
# ---------------------------------------------------------------------------

def query_subject(subject_identifier: str, include_expired: bool = False) -> list[dict]:
    """Art. 15: Return all processing records for a data subject."""
    _ensure_db()
    conn = _get_conn()
    cur = conn.cursor()

    ph = "%s" if _is_postgres() else "?"
    sql = f"""SELECT id, created_at, processing_purpose, lawful_basis, recipients,
                     entity_categories, entity_count, retention_days, expires_at,
                     ai_model_used, risk_level, caller_module, status
              FROM anon_compliance_log
              WHERE subject_identifier = {ph}"""
    if not include_expired:
        sql += " AND status != 'purged'"
    sql += " ORDER BY created_at DESC"

    cur.execute(sql, (subject_identifier,))

    if _is_postgres():
        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            record = dict(zip(cols, row))
            for k, v in record.items():
                if isinstance(v, datetime):
                    record[k] = v.isoformat()
            rows.append(record)
    else:
        rows = [dict(row) for row in cur.fetchall()]

    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Art. 17: Right to Erasure
# ---------------------------------------------------------------------------

def purge_subject(subject_identifier: str) -> int:
    """Art. 17: Erase all records for a data subject. Returns count."""
    _ensure_db()
    conn = _get_conn()
    cur = conn.cursor()
    ph = "%s" if _is_postgres() else "?"

    if _is_postgres():
        cur.execute(
            f"""UPDATE anon_compliance_log
                SET status = 'purged', detail = '{{}}'::jsonb,
                    input_hash = NULL, subject_identifier = NULL
                WHERE subject_identifier = {ph} AND status != 'purged'""",
            (subject_identifier,),
        )
    else:
        cur.execute(
            f"""UPDATE anon_compliance_log
                SET status = 'purged', detail = '{{}}',
                    input_hash = NULL, subject_identifier = NULL
                WHERE subject_identifier = {ph} AND status != 'purged'""",
            (subject_identifier,),
        )
    count = cur.rowcount
    conn.commit()
    conn.close()
    logger.info("DSAR erasure: %d records purged for subject=%s", count, subject_identifier)
    return count


# ---------------------------------------------------------------------------
# Art. 20: Data Portability
# ---------------------------------------------------------------------------

def export_subject(subject_identifier: str) -> str:
    """Art. 20: Export all records as JSON for data portability."""
    records = query_subject(subject_identifier, include_expired=True)
    return json.dumps({
        "data_subject": subject_identifier,
        "exported_at": _now_iso(),
        "record_count": len(records),
        "records": records,
    }, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Retention Purge
# ---------------------------------------------------------------------------

def purge_expired(dry_run: bool = False) -> int:
    """Delete records past retention period. Returns count."""
    _ensure_db()
    conn = _get_conn()
    cur = conn.cursor()

    now = _now_iso() if not _is_postgres() else None

    if dry_run:
        if _is_postgres():
            cur.execute("SELECT COUNT(*) FROM anon_compliance_log WHERE expires_at < NOW() AND status != 'purged'")
        else:
            cur.execute("SELECT COUNT(*) FROM anon_compliance_log WHERE expires_at < ? AND status != 'purged'", (now,))
        count = cur.fetchone()[0] if _is_postgres() else cur.fetchone()["COUNT(*)"]
        conn.close()
        return count

    if _is_postgres():
        cur.execute(
            """UPDATE anon_compliance_log
               SET status = 'purged', detail = '{}'::jsonb,
                   input_hash = NULL, subject_identifier = NULL
               WHERE expires_at < NOW() AND status != 'purged'"""
        )
    else:
        cur.execute(
            """UPDATE anon_compliance_log
               SET status = 'purged', detail = '{}',
                   input_hash = NULL, subject_identifier = NULL
               WHERE expires_at < ? AND status != 'purged'""",
            (now,),
        )
    count = cur.rowcount
    conn.commit()
    conn.close()
    logger.info("Retention purge: %d records purged.", count)
    return count


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary(days: int = 30) -> dict:
    """Processing summary: counts by purpose, basis, recipients, risk level."""
    _ensure_db()
    conn = _get_conn()
    cur = conn.cursor()
    result = {}

    if _is_postgres():
        interval = f"{days} days"
        for dim in ("processing_purpose", "lawful_basis", "risk_level", "status"):
            cur.execute(
                f"SELECT {dim}, COUNT(*) FROM anon_compliance_log "
                "WHERE created_at > NOW() - %s::INTERVAL AND status != 'purged' "
                f"GROUP BY {dim} ORDER BY COUNT(*) DESC",
                (interval,),
            )
            result[dim] = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute(
            """SELECT r, COUNT(*) FROM anon_compliance_log,
               LATERAL unnest(recipients) AS r
               WHERE created_at > NOW() - %s::INTERVAL AND status != 'purged'
               GROUP BY r ORDER BY COUNT(*) DESC""",
            (interval,),
        )
        result["recipients"] = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(entity_count), 0) FROM anon_compliance_log "
            "WHERE created_at > NOW() - %s::INTERVAL AND status != 'purged'",
            (interval,),
        )
        row = cur.fetchone()
        result["total_records"] = row[0]
        result["total_entities_anonymized"] = row[1]
    else:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        for dim in ("processing_purpose", "lawful_basis", "risk_level", "status"):
            cur.execute(
                f"SELECT {dim}, COUNT(*) FROM anon_compliance_log "
                "WHERE created_at > ? AND status != 'purged' "
                f"GROUP BY {dim} ORDER BY COUNT(*) DESC",
                (cutoff,),
            )
            result[dim] = {r[dim]: r["COUNT(*)"] for r in cur.fetchall()}

        # SQLite: recipients is JSON array, need to parse per-row
        cur.execute(
            "SELECT recipients FROM anon_compliance_log "
            "WHERE created_at > ? AND status != 'purged'",
            (cutoff,),
        )
        rcpt_counts = {}
        for r in cur.fetchall():
            for provider in json.loads(r["recipients"]):
                rcpt_counts[provider] = rcpt_counts.get(provider, 0) + 1
        result["recipients"] = rcpt_counts

        cur.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(entity_count), 0) as ent FROM anon_compliance_log "
            "WHERE created_at > ? AND status != 'purged'",
            (cutoff,),
        )
        row = cur.fetchone()
        result["total_records"] = row["cnt"]
        result["total_entities_anonymized"] = row["ent"]

    result["period_days"] = days
    conn.close()
    return result


def ropa_report(days: int = 365) -> str:
    """Record of Processing Activities (GDPR Art. 30, markdown)."""
    s = summary(days)
    lines = [
        "# Record of Processing Activities (ROPA)",
        f"\nPeriod: last {s['period_days']} days",
        f"Total processing records: {s['total_records']}",
        f"Total entities anonymized: {s['total_entities_anonymized']}",
        "\n## Processing Purposes\n",
        "| Purpose | Count |",
        "|---------|-------|",
    ]
    for purpose, count in s.get("processing_purpose", {}).items():
        lines.append(f"| {purpose} | {count} |")

    lines += ["\n## Lawful Bases\n", "| Basis | Count |", "|-------|-------|"]
    for basis, count in s.get("lawful_basis", {}).items():
        lines.append(f"| {basis} | {count} |")

    lines += ["\n## Recipients (Cloud Providers)\n", "| Provider | Count |", "|----------|-------|"]
    for provider, count in s.get("recipients", {}).items():
        lines.append(f"| {provider} | {count} |")

    lines += ["\n## Risk Levels\n", "| Level | Count |", "|-------|-------|"]
    for level, count in s.get("risk_level", {}).items():
        lines.append(f"| {level} | {count} |")

    lines += [
        "\n## Technical Measures",
        "- Anonymization: Local LLM NER + regex safety net",
        "- No raw text stored (SHA-256 hash only for correlation)",
        "- TLS encryption in transit to cloud providers",
        "\n## Data Subject Rights",
        "- Access (Art. 15): `python compliance.py query-subject IDENTIFIER`",
        "- Erasure (Art. 17): `python compliance.py purge-subject IDENTIFIER`",
        "- Portability (Art. 20): `python compliance.py export-subject IDENTIFIER`",
        f"\nGenerated: {_now_iso()}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration hook for anonymize.py
# ---------------------------------------------------------------------------

def enable():
    """Enable compliance logging by setting the env var."""
    os.environ["ANONYMIZE_COMPLIANCE"] = "1"
    _ensure_db()
    logger.info("Compliance logging enabled.")


def is_enabled() -> bool:
    """Check if compliance logging is active."""
    return os.environ.get("ANONYMIZE_COMPLIANCE", "").strip() in ("1", "true", "yes")


def log_pipeline_call(
    text: str,
    provider: str,
    entity_mapping: dict,
    *,
    processing_purpose: str = "general",
    lawful_basis: str = "legitimate_interest",
    risk_level: str = "low",
    subject_identifier: str | None = None,
    consent_reference: str | None = None,
    caller_module: str | None = None,
    retention_days: int | None = None,
    model: str | None = None,
) -> int | None:
    """Log a pipeline() call for GDPR compliance. Called automatically when enabled."""
    if not is_enabled():
        return None

    categories = list({k.strip("[]").rsplit("_", 1)[0] for k in entity_mapping})
    model = model or _DEFAULT_MODEL

    return log_processing(
        processing_purpose=processing_purpose,
        lawful_basis=lawful_basis,
        recipients=[provider],
        entity_categories=categories,
        entity_count=len(entity_mapping),
        retention_days=retention_days,
        ai_model_used=model,
        ai_model_version=get_model_version(model),
        risk_level=risk_level,
        input_hash=hashlib.sha256(text.encode()).hexdigest(),
        subject_identifier=subject_identifier,
        consent_reference=consent_reference,
        caller_module=caller_module,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GDPR & EU AI Act compliance for llm-anonymize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Create/verify DB schema")

    p = sub.add_parser("query-subject", help="Art. 15: query records for a data subject")
    p.add_argument("identifier", help="Subject identifier")
    p.add_argument("--include-expired", action="store_true")

    p = sub.add_parser("purge-subject", help="Art. 17: erase records for a data subject")
    p.add_argument("identifier", help="Subject identifier")

    p = sub.add_parser("export-subject", help="Art. 20: export records as JSON")
    p.add_argument("identifier", help="Subject identifier")

    p = sub.add_parser("purge-expired", help="Purge records past retention period")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("summary", help="Processing summary")
    p.add_argument("--days", type=int, default=30)

    p = sub.add_parser("ropa", help="Full ROPA report (Art. 30)")
    p.add_argument("--days", type=int, default=365)

    args = parser.parse_args()

    if args.command == "setup":
        setup_db()
        print("Compliance DB schema ready.")

    elif args.command == "query-subject":
        records = query_subject(args.identifier, args.include_expired)
        if not records:
            print(f"No records found for: {args.identifier}")
        else:
            print(json.dumps(records, indent=2, ensure_ascii=False, default=str))

    elif args.command == "purge-subject":
        count = purge_subject(args.identifier)
        print(f"Purged {count} records for: {args.identifier}")

    elif args.command == "export-subject":
        print(export_subject(args.identifier))

    elif args.command == "purge-expired":
        count = purge_expired(dry_run=args.dry_run)
        label = "Would purge" if args.dry_run else "Purged"
        print(f"{label} {count} expired records")

    elif args.command == "summary":
        print(json.dumps(summary(args.days), indent=2, default=str))

    elif args.command == "ropa":
        print(ropa_report(args.days))


if __name__ == "__main__":
    main()
