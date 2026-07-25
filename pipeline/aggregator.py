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


def detect(text: str) -> list[Entity]:
    """Runs both detectors and returns the merged result — this is the
    single entry point the rest of the pipeline (masking, UI) should call
    from here on, rather than calling either detector directly."""
    return merge(regex_detect(text), ner_detect(text))


if __name__ == "__main__":
    sample = (
        "Applicant Name: Waida Sehgal\n"
        "Aadhaar Number: 5063 4806 6078\n"
        "PAN Number: CENDE4808R\n"
        "Branch: Ranchi"
    )
    for ent in detect(sample):
        print(ent)