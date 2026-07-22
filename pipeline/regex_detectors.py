"""
regex_detectors.py

Deterministic Aadhaar/PAN detection using regular expressions. This is
one of the two detectors merged later by aggregator.py — it has no
knowledge of the NER detector at all, which is what keeps it
independently testable.
"""

import re
from dataclasses import dataclass


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
    Basic sanity checks — not the full UIDAI checksum, just enough to
    cut obvious false positives:
    - reject a leading 0 or 1 (UIDAI never issues these)
    - reject a sequence of all-identical digits
    """
    digits = candidate.replace(" ", "")
    if digits[0] in "01":
        return False
    if len(set(digits)) == 1:
        return False
    return True


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