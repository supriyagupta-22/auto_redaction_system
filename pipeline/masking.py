"""
masking.py

Applies redaction to detected entities in one of two modes:
  - "redact"       : replaces every entity with a fixed [REDACTED] mask
  - "pseudonymize" : replaces every entity with a short, consistent
                     token derived from a keyed hash of its value

Pseudonymization uses HMAC-SHA256 with a locally-stored secret key, so
the same input value always produces the same token — this consistency
comes purely from the hash being deterministic, not from any database
lookup. The original value itself is never stored anywhere, even in
hashed form; HMAC with a secret key is a one-way function, so the
token cannot be reversed back into the original value without the key.
"""

import hashlib
import hmac
import os
from pathlib import Path

from pipeline.regex_detectors import Entity

SECRET_KEY_PATH = Path(__file__).parent.parent / "db" / "secret.key"


def _get_secret_key() -> bytes:
    """
    Loads the local secret key used to key the HMAC, generating one on
    first run if it doesn't exist yet. This file must stay local and
    must NEVER be committed to git (add db/secret.key to .gitignore) —
    if it ever changed or leaked, every pseudonym token would either
    become inconsistent or become reversible by whoever has the key.
    """
    if not SECRET_KEY_PATH.exists():
        SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECRET_KEY_PATH.write_bytes(os.urandom(32))
    return SECRET_KEY_PATH.read_bytes()


def pseudonymize(value: str, entity_type: str) -> str:
    """
    Returns a short, consistent token for a given value, e.g.
    'NAME_4F2A1C'. Same value + same local key always produces the
    same token; different values essentially never collide.
    """
    key = _get_secret_key()
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{entity_type}_{digest[:6].upper()}"


def mask(text: str, entities: list[Entity], mode: str = "redact") -> str:
    """
    Replaces every entity span in `text` with either a fixed mask or a
    pseudonymized token, depending on `mode`.

    Processes entities from the END of the text backwards (highest
    start offset first). This matters: replacing an earlier span would
    shift every later offset, corrupting them. Working backwards means
    every offset we haven't processed yet is still valid at the moment
    we use it, since nothing before it has been touched.
    """
    if mode not in ("redact", "pseudonymize"):
        raise ValueError(f"Unknown mode: {mode!r}")

    result = text
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        if mode == "redact":
            replacement = "[REDACTED]"
        else:
            replacement = f"[{pseudonymize(entity.text, entity.label)}]"
        result = result[:entity.start] + replacement + result[entity.end:]

    return result


if __name__ == "__main__":
    from pipeline.aggregator import detect

    sample = (
        "Applicant Name: Waida Sehgal\n"
        "Aadhaar Number: 2345 6789 0124\n"
        "PAN Number: ABCPD1234E\n"
        "Branch: Ranchi"
    )
    entities = detect(sample)

    print("--- Redacted ---")
    print(mask(sample, entities, mode="redact"))
    print("\n--- Pseudonymized ---")
    print(mask(sample, entities, mode="pseudonymize"))