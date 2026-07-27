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


_VALID_PAN_HOLDER_TYPES = set("ABCFGHLJPT")


def _validate_pan(candidate: str) -> bool:
    """
    Checks the 4th character against the fixed set of PAN holder-type
    codes published by the Income Tax Department (P=individual,
    C=company, H=HUF, F=firm, A=association, T=trust, B=body of
    individuals, L=local authority, J=artificial juridical person,
    G=government). A purely random string only has a ~38% (10/26)
    chance of landing on one of these ten letters, so this meaningfully
    cuts false positives beyond the format regex alone.

    Note: PAN's final character also functions as a check digit in
    practice, but the Income Tax Department has not publicly documented
    the checksum formula the way UIDAI has for Aadhaar's Verhoeff
    digit — validating against an unverified guessed formula would be
    worse than not validating it at all, so it's deliberately left out.
    """
    return candidate[3] in _VALID_PAN_HOLDER_TYPES


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
        "Aadhaar Number: 2345 6789 0124\n"
        "PAN Number: ABCPD1234E\n"
        "Branch: Ranchi"
    )
    for ent in detect(sample):
        print(ent)