"""
tests/test_pipeline.py

Checks the regex detector's recall against the full synthetic labeled
corpus, not just a single example. This only checks AADHAAR/PAN —
NAME/LOCATION detection is the NER detector's job and gets evaluated
once that module exists.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.regex_detectors import detect_pan, detect_aadhaar

LABELS_PATH = Path(__file__).parent.parent / "data" / "synthetic_corpus" / "labels.jsonl"


def run_check():
    total_pan, total_aadhaar = 0, 0
    found_pan, found_aadhaar = 0, 0

    with open(LABELS_PATH) as f:
        for line in f:
            record = json.loads(line)
            text = record["text"]

            gold_pan = {(e["start"], e["end"]) for e in record["entities"] if e["label"] == "PAN"}
            gold_aadhaar = {(e["start"], e["end"]) for e in record["entities"] if e["label"] == "AADHAAR"}

            pred_pan = {(e.start, e.end) for e in detect_pan(text)}
            pred_aadhaar = {(e.start, e.end) for e in detect_aadhaar(text)}

            total_pan += len(gold_pan)
            total_aadhaar += len(gold_aadhaar)
            found_pan += len(gold_pan & pred_pan)
            found_aadhaar += len(gold_aadhaar & pred_aadhaar)

    print(f"PAN recall:      {found_pan}/{total_pan} ({found_pan/total_pan:.1%})")
    print(f"Aadhaar recall:  {found_aadhaar}/{total_aadhaar} ({found_aadhaar/total_aadhaar:.1%})")


if __name__ == "__main__":
    run_check()