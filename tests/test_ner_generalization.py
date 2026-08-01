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
    "KYC Verification Letter",                # regression test: this exact phrase was
                                               # once wrongly tagged as a NAME (see
                                               # generate_synthetic.py templates 9-10)
    "Please find attached the Final Notice regarding your account.",  # was wrongly
                                               # tagged LOCATION before templates 11-12
    "Kindly review the Escalation Matrix before proceeding.",  # genuinely new phrase,
                                               # deliberately excluded from the distractor
                                               # pool — tests whether the broader,
                                               # systematic fix actually generalizes
]

for sentence in NOVEL_SENTENCES:
    entities = detect(sentence)
    print(f"\n{sentence!r}")
    if not entities:
        print("  -> no entities detected")
    for ent in entities:
        print(f"  -> {ent.label} : {ent.text!r}")