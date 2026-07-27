"""
tests/test_false_positives.py

Checks that the regex detectors correctly REJECT things that merely
look like Aadhaar/PAN numbers but aren't — the precision half of
testing that test_pipeline.py (recall-only) doesn't cover.

Every value below is deliberately chosen and verified to fail the
relevant validator: the two Aadhaar-shaped numbers have a valid
leading digit but a failing Verhoeff checksum, and the two PAN-shaped
strings have a 4th character that isn't a valid holder-type code.

Run:
    python tests/test_false_positives.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.regex_detectors import detect_aadhaar, detect_pan

NEGATIVE_CASES = [
    "Order ID: 9876 5432 1098",         # Aadhaar-shaped, fails Verhoeff
    "Tracking number: 4567 8901 2345",  # Aadhaar-shaped, fails Verhoeff
    "Reference code: XXKKD1234Z",       # PAN-shaped, 'K' isn't a valid holder-type code
    "Invoice number: ZZQQR5678W",       # PAN-shaped, 'Q' isn't a valid holder-type code
    "The meeting has been rescheduled to next week.",  # no PII at all
]


def run_check():
    failures = 0

    for text in NEGATIVE_CASES:
        hits = detect_aadhaar(text) + detect_pan(text)
        if hits:
            failures += 1
            print(f"FALSE POSITIVE on: {text!r}")
            for hit in hits:
                print(f"  -> wrongly flagged: {hit}")
        else:
            print(f"OK (correctly rejected): {text!r}")

    print(f"\n{len(NEGATIVE_CASES) - failures}/{len(NEGATIVE_CASES)} negative cases correctly rejected")


if __name__ == "__main__":
    run_check()