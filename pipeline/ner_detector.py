"""
ner_detector.py

Loads the fine-tuned spaCy NER model and exposes a detect() function
matching the same interface as regex_detectors.detect() — a list of
Entity objects — so the Aggregator can treat both detectors identically.
"""

from pathlib import Path

import spacy

from pipeline.regex_detectors import Entity  # reuse the same Entity shape

MODEL_PATH = Path(__file__).parent.parent / "models" / "ner_model" / "model-best"

_nlp = None  # loaded lazily so just importing this module doesn't load the model


def _get_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(MODEL_PATH)
    return _nlp


def detect(text: str) -> list[Entity]:
    nlp = _get_model()
    doc = nlp(text)
    return [Entity(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents]


if __name__ == "__main__":
    sample = "Dear Rohan Mehta, please visit our branch in Nashik for further assistance."
    for ent in detect(sample):
        print(ent)