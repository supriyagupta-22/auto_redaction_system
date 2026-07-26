"""
regex_detectors.py

Deterministic Aadhaar/PAN detection using regular expressions. This is
one of the two detectors merged later by aggregator.py — it has no
knowledge of the NER detector at all, which is what keeps it
independently testable.
"""

import re
from dataclasses import dataclass

from pipeline.verhoeff import is_valid as _verhoeff_is_valid


@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str


PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def _validate_aadhaar(candidate: str) -> bool:
    """
    Full Verhoeff checksum validation — see verhoeff.py. This replaces
    the earlier basic sanity checks (leading digit, all-identical
    digits) with the actual algorithm UIDAI uses, which rejects the
    large majority of unrelated 12-digit numbers (phone sequences,
    order IDs, etc.) that happen to match the format but were never
    generated with a valid check digit.
    """
    digits = candidate.replace(" ", "")
    if digits[0] in "01":
        return False
    return _verhoeff_is_valid(digits)


def _validate_pan(candidate: str) -> bool:
    """
    PAN's structure is already fully constrained by the regex (5
    letters, 4 digits, 1 letter), so there's little extra to check.
    This function exists mainly as a place for that validation logic
    to grow later — e.g. checking the 4th character against known
    PAN holder-type codes (P = individual, C = company, etc.).
    """
    return True


def detect_pan(text: str) -> list[Entity]:
    entities = []
    for match in PAN_PATTERN.finditer(text):
        candidate = match.group()
        if _validate_pan(candidate):
            entities.append(Entity(match.start(), match.end(), "PAN", candidate))
    return entities


def detect_aadhaar(text: str) -> list[Entity]:
    entities = []
    for match in AADHAAR_PATTERN.finditer(text):
        candidate = match.group()
        if _validate_aadhaar(candidate):
            entities.append(Entity(match.start(), match.end(), "AADHAAR", candidate))
    return entities


def detect(text: str) -> list[Entity]:
    """Runs both detectors and returns one combined, offset-sorted list."""
    entities = detect_pan(text) + detect_aadhaar(text)
    entities.sort(key=lambda e: e.start)
    return entities


if __name__ == "__main__":
    # quick manual smoke test
    sample = (
        "Applicant Name: Waida Sehgal\n"
        "Aadhaar Number: 5063 4806 6078\n"
        "PAN Number: CENDE4808R\n"
        "Branch: Ranchi"
    )
    for ent in detect(sample):
        print(ent)