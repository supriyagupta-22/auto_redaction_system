"""
aggregator.py

Merges the regex detector's output (Aadhaar, PAN) with the NER
detector's output (Name, Location) into one sorted, non-overlapping
list of entities.

Precedence rule: when a regex match and an NER match overlap, the
regex match wins — it's deterministic and format-validated, so it's
more trustworthy than a probabilistic NER guess covering the same
text. Non-overlapping entities from both detectors are simply merged.
"""

from pipeline.ner_detector import detect as ner_detect
from pipeline.regex_detectors import Entity
from pipeline.regex_detectors import detect as regex_detect


def _overlaps(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


def merge(regex_entities: list[Entity], ner_entities: list[Entity]) -> list[Entity]:
    """Combines two entity lists, resolving overlaps in favor of regex
    matches. Returns one list, sorted by start offset."""
    merged = list(regex_entities)  # regex entities are always kept as-is

    for ner_ent in ner_entities:
        if not any(_overlaps(ner_ent, regex_ent) for regex_ent in regex_entities):
            merged.append(ner_ent)

    merged.sort(key=lambda e: e.start)
    return merged


def filter_by_labels(entities: list[Entity], allowed_labels: set) -> list[Entity]:
    """Returns only the entities whose label is in allowed_labels.
    Used to implement per-category redaction toggles in the UI — e.g.
    a user who wants Aadhaar/PAN masked but not names. Entities not in
    allowed_labels are dropped entirely, not just left unmasked — they
    won't appear in the redacted output OR the audit log, since
    nothing was actually done to them."""
    return [e for e in entities if e.label in allowed_labels]


def detect(text: str) -> list[Entity]:
    """Runs both detectors and returns the merged result — this is the
    single entry point the rest of the pipeline (masking, UI) should call
    from here on, rather than calling either detector directly."""
    return merge(regex_detect(text), ner_detect(text))


def filter_by_category(entities: list[Entity], enabled_labels: set) -> list[Entity]:
    """Returns only the entities whose label is in enabled_labels,
    preserving order. Lets the UI selectively enable/disable redaction
    categories per run without needing to re-run detection — detection
    always finds everything; this decides what actually gets acted on."""
    return [e for e in entities if e.label in enabled_labels]


if __name__ == "__main__":
    sample = (
        "Applicant Name: Waida Sehgal\n"
        "Aadhaar Number: 2345 6789 0124\n"
        "PAN Number: ABCPD1234E\n"
        "Branch: Ranchi"
    )
    for ent in detect(sample):
        print(ent)