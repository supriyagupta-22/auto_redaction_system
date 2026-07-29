"""
check_novel_value_overlap.py

Checks a batch of CANDIDATE names/locations against the training
corpus, so we can pick out ones that are confirmed genuinely novel
before building generalization test sentences around them — rather
than guessing and finding out after the fact, the way Nashik slipped
through last time.

Run:
    python check_novel_value_overlap.py
"""

import json
from pathlib import Path

LABELS_PATH = Path("data/synthetic_corpus/labels.jsonl")

CANDIDATE_NAMES = [
    "Ananya Krishnan",
    "Vikram Desai",
]

CANDIDATE_LOCATIONS = [
    "Ratlam",
    "Bharuch",
    "Karimnagar",
    "Sivakasi",
    "Hosur",
    "Bettiah",
    "Chandrapur",
    "Palakkad",
    "Thoothukudi",
    "Kolhapur",
]

seen_names = set()
seen_locations = set()

with open(LABELS_PATH, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        text = record["text"]
        for ent in record["entities"]:
            value = text[ent["start"]:ent["end"]]
            if ent["label"] == "NAME":
                seen_names.add(value)
            elif ent["label"] == "LOCATION":
                seen_locations.add(value)

print("=== NAME candidates ===")
for name in CANDIDATE_NAMES:
    status = "SEEN during training" if name in seen_names else "genuinely novel"
    print(f"{name!r}: {status}")

print("\n=== LOCATION candidates ===")
for loc in CANDIDATE_LOCATIONS:
    status = "SEEN during training" if loc in seen_locations else "genuinely novel"
    print(f"{loc!r}: {status}")

print(f"\nTotal unique NAME values in training corpus: {len(seen_names)}")
print(f"Total unique LOCATION values in training corpus: {len(seen_locations)}")