"""
verhoeff.py

Implementation of the Verhoeff checksum algorithm — the same
error-detecting check-digit scheme UIDAI uses for the 12th digit of a
real Aadhaar number. Used two ways in this project:

  - regex_detectors.py uses is_valid() to reject 12-digit sequences
    that can't possibly be a real Aadhaar number, cutting down false
    positives from unrelated 12-digit numbers (phone sequences, order
    IDs, etc.) that happen to match the basic format.
  - generate_synthetic.py uses generate_check_digit() so synthetic
    training data contains Aadhaar-shaped numbers that actually pass
    this same validation, instead of pure random digits that would
    fail it almost every time.

Note: passing this checksum only proves a number is *shaped* like a
valid Aadhaar number — it does not confirm the number actually exists
or belongs to anyone. Only UIDAI's own database can do that.
"""

_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def is_valid(number_str: str) -> bool:
    """Returns True if the full number (including its final check
    digit) satisfies the Verhoeff checksum."""
    c = 0
    for i, ch in enumerate(reversed(number_str)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def generate_check_digit(number_str: str) -> str:
    """Given all digits EXCEPT the check digit, returns the correct
    check digit to append so the resulting full number passes
    is_valid()."""
    c = 0
    for i, ch in enumerate(reversed(number_str)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])