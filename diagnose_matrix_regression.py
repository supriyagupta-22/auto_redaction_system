"""
diagnose_matrix_regression.py

Tests whether the "Matrix" false positive is an isolated case or part
of a broader pattern, by trying several similar generic business-
jargon nouns across a few different sentence structures.

Run from the project root:
    python diagnose_matrix_regression.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pipeline.ner_detector import detect

CANDIDATE_WORDS = [
    "Matrix", "Portal", "Dashboard", "System", "Module",
    "Framework", "Gateway", "Protocol", "Directory", "Registry",
]

SENTENCE_TEMPLATES = [
    "Kindly review the Escalation {word} before proceeding.",
    "Please check the {word} for further information.",
    "The {word} was updated yesterday.",
]

for template in SENTENCE_TEMPLATES:
    print(f"=== Template: {template!r} ===")
    for word in CANDIDATE_WORDS:
        sentence = template.format(word=word)
        entities = detect(sentence)
        status = "; ".join(f"{e.label}:{e.text!r}" for e in entities) if entities else "clean"
        print(f"  {word:12s} -> {status}")
    print()

print("=== Round 2 specific regressions ===")
extra_checks = [
    "Our new branch in Coimbatore opens next month.",
    "Please confirm the Aacrawr details before submission.",
]
for sentence in extra_checks:
    entities = detect(sentence)
    status = "; ".join(f"{e.label}:{e.text!r}" for e in entities) if entities else "clean"
    print(f"  {sentence!r} -> {status}")