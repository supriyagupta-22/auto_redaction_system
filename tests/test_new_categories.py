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
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+91[\-\s]?|0)?[6-9]\d{9}\b")
GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d\b")
IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
CARD_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,4}\b")
PASSPORT_PATTERN = re.compile(r"\b[A-Z]\d{7}\b")


def _validate_aadhaar(candidate: str) -> bool:
    """
    Full Verhoeff checksum validation — see verhoeff.py. This replaces
    the earlier basic sanity checks (leading digit, all-identical
    digits) with the actual algorithm UIDAI uses, which rejects the
    large majority of unrelated 12-digit numbers (phone sequences,
    order IDs, etc.) that happen to match the format but were never
    generated with a valid check digit.
    """
    digits = re.sub(r"\s", "", candidate)
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


def detect_email(text: str) -> list[Entity]:
    return [Entity(m.start(), m.end(), "EMAIL", m.group()) for m in EMAIL_PATTERN.finditer(text)]


def detect_phone(text: str) -> list[Entity]:
    # The regex itself already constrains the leading digit to 6-9 and
    # the length to exactly 10 — no separate validator needed, unlike
    # PAN/Aadhaar where format and validity are genuinely different checks.
    return [Entity(m.start(), m.end(), "PHONE", m.group()) for m in PHONE_PATTERN.finditer(text)]


def detect_gstin(text: str) -> list[Entity]:
    # Format-only, like PAN — GSTIN does have an official checksum
    # digit too, but it's not published the way Verhoeff is for
    # Aadhaar, so it isn't implemented here (same honest reasoning as
    # PAN's un-checked final letter).
    return [Entity(m.start(), m.end(), "GSTIN", m.group()) for m in GSTIN_PATTERN.finditer(text)]


def detect_ifsc(text: str) -> list[Entity]:
    return [Entity(m.start(), m.end(), "IFSC", m.group()) for m in IFSC_PATTERN.finditer(text)]


def _validate_card(candidate: str) -> bool:
    """Luhn algorithm — the standard checksum used by essentially all
    major card networks (Visa, Mastercard, Amex, etc.) to catch
    accidental digit errors. Plays the same role here that Verhoeff
    plays for Aadhaar: filters out random digit sequences that happen
    to match the length/format but were never a real generated card
    number."""
    digits = re.sub(r"[\s-]", "", candidate)
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect_card(text: str) -> list[Entity]:
    entities = []
    for match in CARD_PATTERN.finditer(text):
        candidate = match.group()
        if _validate_card(candidate):
            entities.append(Entity(match.start(), match.end(), "CARD", candidate))
    return entities


def detect_passport(text: str) -> list[Entity]:
    # Format-only. Indian passport numbers have no publicly documented
    # checksum at all (unlike Aadhaar or card numbers), so this is
    # genuinely the weakest of all the detectors here — flagged
    # honestly as a known limitation rather than implied to be as
    # reliable as the checksum-backed ones.
    return [Entity(m.start(), m.end(), "PASSPORT", m.group()) for m in PASSPORT_PATTERN.finditer(text)]


def _overlaps(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


def detect(text: str) -> list[Entity]:
    """Runs every regex detector and returns one combined,
    offset-sorted, non-overlapping list. If two patterns ever produce
    overlapping spans (rare in practice, since every pattern here uses
    \\b word boundaries that already prevent most embedded-substring
    collisions), the longer/more specific match wins — kept as a
    defensive safety net rather than because it's frequently needed."""
    all_entities = (
        detect_pan(text)
        + detect_aadhaar(text)
        + detect_email(text)
        + detect_phone(text)
        + detect_gstin(text)
        + detect_ifsc(text)
        + detect_card(text)
        + detect_passport(text)
    )
    all_entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

    resolved = []
    for entity in all_entities:
        if not any(_overlaps(entity, kept) for kept in resolved):
            resolved.append(entity)

    resolved.sort(key=lambda e: e.start)
    return resolved


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