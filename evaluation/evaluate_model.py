"""
evaluation/evaluate_model.py

Formal precision/recall/F1 evaluation of the full detection pipeline
(regex + NER combined, via aggregator.detect()) against the held-out
test split, using spaCy's built-in Scorer — the standard, citable
methodology for NER evaluation, rather than hand-rolled span-matching
logic like the earlier test scripts used.

Scope: covers AADHAAR, PAN, NAME, LOCATION — the four categories with
real ground-truth labels in the synthetic corpus (recorded during
generation in labels.jsonl). The six categories added later (Email,
Phone, GSTIN, IFSC, Card, Passport) are deterministic format/checksum
matches without natural-language ambiguity, and continue to be
evaluated via the targeted, hand-verified cases in
tests/test_new_categories.py — large-scale statistical evaluation
doesn't add much rigor for a fixed-format regex match the way it does
for context-dependent NER.

Note on interpretation: this measures in-distribution performance —
the test split comes from the same synthetic generation process as
training. It answers a different question than
tests/test_ner_generalization.py (which deliberately uses hand-written,
out-of-distribution sentences). Both are valid and worth citing
together in Chapter 6 — this one for a standard, statistically
grounded per-category score; that one for genuine generalization
evidence.

Run from the project root:
    python evaluation/evaluate_model.py
"""

import csv
import json
import sys
from pathlib import Path

import spacy
from spacy.scorer import Scorer
from spacy.training import Example

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.aggregator import detect as pipeline_detect

LABELS_PATH = Path(__file__).parent.parent / "data" / "synthetic_corpus" / "labels.jsonl"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.csv"

# A blank tokenizer used only to build Doc objects for scoring — not
# the trained NER model itself, which is called indirectly via
# pipeline_detect(). Using the same blank tokenizer for both gold and
# predicted Docs keeps token alignment consistent between them.
NLP_BLANK = spacy.blank("en")


def load_test_records():
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] == "test":
                yield record


def build_examples():
    examples = []
    skipped_gold, skipped_pred = 0, 0

    for record in load_test_records():
        text = record["text"]

        # Gold document: entities exactly as recorded during generation.
        gold_doc = NLP_BLANK.make_doc(text)
        gold_spans = []
        for ent in record["entities"]:
            span = gold_doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is None:
                skipped_gold += 1
                continue
            gold_spans.append(span)
        gold_doc.ents = gold_spans

        # Predicted document: run the REAL pipeline (regex + NER
        # merged via the aggregator) so this evaluates what the system
        # actually does end-to-end, not the NER model in isolation.
        pred_doc = NLP_BLANK.make_doc(text)
        pred_spans = []
        for ent in pipeline_detect(text):
            span = pred_doc.char_span(ent.start, ent.end, label=ent.label)
            if span is None:
                skipped_pred += 1
                continue
            pred_spans.append(span)
        pred_doc.ents = pred_spans

        examples.append(Example(pred_doc, gold_doc))

    if skipped_gold or skipped_pred:
        print(f"(skipped {skipped_gold} gold / {skipped_pred} predicted spans that "
              f"didn't align to token boundaries)")

    return examples


def main():
    examples = build_examples()
    print(f"Evaluating on {len(examples)} held-out test documents\n")

    scorer = Scorer()
    scores = scorer.score(examples)

    rows = [("OVERALL", scores["ents_p"], scores["ents_r"], scores["ents_f"])]
    for label, metrics in sorted(scores["ents_per_type"].items()):
        rows.append((label, metrics["p"], metrics["r"], metrics["f"]))

    print(f"{'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for label, p, r, f in rows:
        print(f"{label:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["category", "precision", "recall", "f1"])
        for label, p, r, f in rows:
            writer.writerow([label, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"])

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()