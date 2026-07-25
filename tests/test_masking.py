"""
tests/test_masking.py

Unit tests for the masking engine, using hand-built Entity objects.
The multi-entity test is the important one — it checks that replacing
several spans in one string doesn't corrupt the offsets of entities
that haven't been processed yet.

Run:
    python tests/test_masking.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.masking import mask, pseudonymize
from pipeline.regex_detectors import Entity


def test_redact_single_entity():
    text = "My name is Waida."
    entities = [Entity(11, 16, "NAME", "Waida")]
    result = mask(text, entities, mode="redact")
    assert result == "My name is [REDACTED]."
    print("test_redact_single_entity: PASSED")


def test_redact_multiple_entities_preserves_positions():
    text = "Name: Waida, City: Ranchi."
    entities = [
        Entity(6, 11, "NAME", "Waida"),
        Entity(19, 25, "LOCATION", "Ranchi"),
    ]
    result = mask(text, entities, mode="redact")
    assert result == "Name: [REDACTED], City: [REDACTED]."
    print("test_redact_multiple_entities_preserves_positions: PASSED")


def test_pseudonymize_is_consistent():
    token1 = pseudonymize("Waida Sehgal", "NAME")
    token2 = pseudonymize("Waida Sehgal", "NAME")
    assert token1 == token2
    print("test_pseudonymize_is_consistent: PASSED")


def test_pseudonymize_differs_by_value():
    token1 = pseudonymize("Waida Sehgal", "NAME")
    token2 = pseudonymize("Rohan Mehta", "NAME")
    assert token1 != token2
    print("test_pseudonymize_differs_by_value: PASSED")


if __name__ == "__main__":
    test_redact_single_entity()
    test_redact_multiple_entities_preserves_positions()
    test_pseudonymize_is_consistent()
    test_pseudonymize_differs_by_value()
    print("\nAll masking tests passed.")