"""
tests/test_aggregator.py

Unit tests for the aggregator's merge logic using hand-built Entity
objects, not real detector output — this isolates aggregator.py from
whatever the regex/NER detectors currently do, so a failure here
always points to the merge logic itself.

Run:
    python tests/test_aggregator.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.aggregator import merge
from pipeline.regex_detectors import Entity


def test_no_overlap_keeps_both():
    regex_entities = [Entity(0, 10, "PAN", "CENDE4808R")]
    ner_entities = [Entity(20, 30, "NAME", "Waida Sehgal")]
    result = merge(regex_entities, ner_entities)
    assert len(result) == 2
    print("test_no_overlap_keeps_both: PASSED")


def test_overlap_regex_wins():
    regex_entities = [Entity(5, 15, "PAN", "CENDE4808R")]
    ner_entities = [Entity(8, 20, "NAME", "some overlapping guess")]
    result = merge(regex_entities, ner_entities)
    assert len(result) == 1
    assert result[0].label == "PAN"
    print("test_overlap_regex_wins: PASSED")


def test_result_is_sorted():
    regex_entities = [Entity(50, 60, "PAN", "x")]
    ner_entities = [Entity(0, 10, "NAME", "y")]
    result = merge(regex_entities, ner_entities)
    assert result[0].start == 0
    assert result[1].start == 50
    print("test_result_is_sorted: PASSED")


if __name__ == "__main__":
    test_no_overlap_keeps_both()
    test_overlap_regex_wins()
    test_result_is_sorted()
    print("\nAll aggregator tests passed.")