"""
db_logger.py

Handles all SQLite persistence: creates the schema on first run, and
logs every processed document plus every entity redacted in it. This
is what the risk dashboard and CSV export (Roadmap Phase 7) will read
from later — nothing downstream should write to the database except
through this module.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.masking import pseudonymize
from pipeline.regex_detectors import Entity

DB_PATH = Path(__file__).parent.parent / "db" / "redaction_log.sqlite"

# Regex-detected entities are fully deterministic, so a confidence of
# 1.0 is genuinely true for them, not a placeholder. NER-derived
# entities don't currently expose a real per-entity confidence score
# through spaCy's standard .ents output, so NULL is logged rather than
# inventing a number that was never actually computed.
REGEX_LABELS = {"AADHAAR", "PAN"}

SEVERITY_MAP = {
    "AADHAAR": "CRITICAL",
    "PAN": "CRITICAL",
    "NAME": "HIGH",
    "LOCATION": "LOW",
}


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates the schema if it doesn't already exist. Safe to call on
    every app startup — CREATE TABLE IF NOT EXISTS is a no-op once the
    tables already exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_time TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS redaction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            entity_type TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            replacement_token TEXT NOT NULL,
            confidence_score REAL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );

        CREATE TABLE IF NOT EXISTS pseudonym_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_hash TEXT NOT NULL UNIQUE,
            token TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _hash_value(value: str) -> str:
    """A plain (non-keyed) hash used only as an internal lookup value
    in the audit log — separate from the keyed HMAC pseudonym tokens
    in masking.py. This one-way hash lets you spot 'is this the same
    original value as another row' without ever storing or being able
    to recover the original value itself."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def log_document(filename: str, file_type: str) -> int:
    """Inserts one row into `documents` and returns its new id, used
    as the `document_id` for every entity found in it."""
    conn = _get_connection()
    cursor = conn.execute(
        "INSERT INTO documents (filename, file_type, upload_time) VALUES (?, ?, ?)",
        (filename, file_type, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    document_id = cursor.lastrowid
    conn.close()
    return document_id


def log_entities(document_id: int, entities: list[Entity], mode: str = "redact"):
    """Logs one row per detected entity for this document. `mode`
    mirrors masking.mask()'s own mode parameter, so the token recorded
    here always matches what was actually written into the output."""
    conn = _get_connection()
    timestamp = datetime.now(timezone.utc).isoformat()

    for entity in entities:
        confidence = 1.0 if entity.label in REGEX_LABELS else None
        severity = SEVERITY_MAP.get(entity.label, "LOW")
        original_hash = _hash_value(entity.text)

        if mode == "redact":
            token = "[REDACTED]"
        else:
            token = f"[{pseudonymize(entity.text, entity.label)}]"

        conn.execute(
            """INSERT INTO redaction_events
               (document_id, entity_type, original_hash, replacement_token,
                confidence_score, severity, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (document_id, entity.label, original_hash, token, confidence, severity, timestamp),
        )

        if mode == "pseudonymize":
            # INSERT OR IGNORE: if this exact original value has been
            # pseudonymized before, silently skip — this is what makes
            # a later `SELECT COUNT(*) FROM pseudonym_map` a genuine
            # count of unique individuals/values redacted, not a count
            # of every occurrence across every document.
            conn.execute(
                """INSERT OR IGNORE INTO pseudonym_map
                   (original_hash, token, created_at)
                   VALUES (?, ?, ?)""",
                (original_hash, token, timestamp),
            )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    from pipeline.aggregator import detect

    init_db()

    sample = (
        "Applicant Name: Waida Sehgal\n"
        "Aadhaar Number: 2345 6789 0124\n"
        "PAN Number: ABCPD1234E\n"
        "Branch: Ranchi"
    )
    entities = detect(sample)

    doc_id_redact = log_document("sample_redact.txt", "txt")
    log_entities(doc_id_redact, entities, mode="redact")

    # Also log once in pseudonymize mode, so pseudonym_map actually gets
    # exercised rather than sitting empty and unverified.
    doc_id_pseudo = log_document("sample_pseudonymize.txt", "txt")
    log_entities(doc_id_pseudo, entities, mode="pseudonymize")

    conn = _get_connection()
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_events = conn.execute("SELECT COUNT(*) FROM redaction_events").fetchone()[0]
    total_pseudonyms = conn.execute("SELECT COUNT(*) FROM pseudonym_map").fetchone()[0]
    conn.close()

    print(f"Logged document {doc_id_redact} (redact mode) and {doc_id_pseudo} (pseudonymize mode).")
    print(f"Total documents in DB so far: {total_docs}")
    print(f"Total redaction events in DB so far: {total_events}")
    print(f"Total unique pseudonym mappings in DB so far: {total_pseudonyms}")