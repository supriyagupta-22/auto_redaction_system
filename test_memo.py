"""
test_memo.py

One-off diagnostic: runs the full pipeline (regex + NER via the
aggregator) against memo.txt and prints every detected entity clearly,
grouped by label for easy comparison against the source document.

Run from the project root, with memo.txt also in the project root:
    python test_memo.py
"""

from pipeline.aggregator import detect

with open("memo.txt", encoding="utf-8") as f:
    text = f.read()

entities = detect(text)
print(f"Total entities detected: {len(entities)}\n")

for e in sorted(entities, key=lambda e: e.label):
    print(f"{e.label:10s} {e.text!r}")