"""
tests/test_ner_generalization.py

Checks whether the NER model learned genuine patterns or just
memorized the four training templates. Every sentence below uses a
structure that does NOT appear in any training template — different
verbs, different entity position, different surrounding words.

This is not a formal precision/recall evaluation (that's Roadmap
Phase 9, against a proper held-out set) — it's a fast, human-readable
gut check to run right after training, before building anything else
on top of a model that might not actually generalize.

Run:
    python tests/test_ner_generalization.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.ner_detector import detect

NOVEL_SENTENCES = [
    "Rohan Mehta submitted his documents yesterday.",
    "The Nashik office will remain closed on Monday.",
    "Please forward this email to Priya Nair before Friday.",
    "Our new branch in Coimbatore opens next month.",
    "The weather today is quite pleasant.",   # no PII — checks for false positives
    "He said the meeting went well.",         # no PII — checks for false positives
    "The Mysore division is expanding rapidly.",
    "We need to send the report to the Trivandrum team.",
    "Flights to Guwahati are delayed.",
    "Vikram Desai will be joining the project tomorrow.",
    "The package was shipped to Ananya Krishnan.",
    "Our facility in Bharuch is fully operational.",
    "The audit at the Sivakasi plant concluded.",
    "Palakkad authorities have approved the request."
]

for sentence in NOVEL_SENTENCES:
    entities = detect(sentence)
    print(f"\n{sentence!r}")
    if not entities:
        print("  -> no entities detected")
    for ent in entities:
        print(f"  -> {ent.label} : {ent.text!r}")