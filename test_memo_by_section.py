"""
test_memo_by_section.py

Splits memo.txt into its 6 individual sections and runs each through
the pipeline separately — tests whether NAME/LOCATION quality recovers
on shorter, simpler chunks of the SAME real document, which would
confirm document length/structural complexity (not word content) as
the actual cause of the degraded results on the full memo.

Run from the project root, with memo.txt also in the project root:
    python test_memo_by_section.py
"""

import re

from pipeline.aggregator import detect

with open("memo.txt", encoding="utf-8") as f:
    full_text = f.read()

# Split on the "SECTION N:" headers
sections = re.split(r"(?=SECTION \d+:)", full_text)

for section in sections:
    if not section.strip():
        continue
    header = section.strip().split("\n")[0]
    entities = [e for e in detect(section) if e.label in ("NAME", "LOCATION")]
    print(f"=== {header} ===")
    print(f"  {len(section)} characters, {len(entities)} NAME/LOCATION entities")
    for e in entities:
        print(f"    {e.label:10s} {e.text!r}")
    print()