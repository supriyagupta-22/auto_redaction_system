"""
prepare_training_data.py

Converts data/synthetic_corpus/labels.jsonl into spaCy's binary .spacy
training format, split into a train set and a dev set (spaCy's term
for a validation set used during training).

Only NAME and LOCATION entities are included — AADHAAR/PAN are
deliberately excluded, since the regex detector already handles those
deterministically and there's no reason to make the NER model re-learn
a fixed alphanumeric pattern.

Output:
    data/synthetic_corpus/train.spacy
    data/synthetic_corpus/dev.spacy

Run:
    python data/prepare_training_data.py
"""

import json
from pathlib import Path

import spacy
from spacy.tokens import DocBin

LABELS_PATH = Path(__file__).parent / "synthetic_corpus" / "labels.jsonl"
OUTPUT_DIR = Path(__file__).parent / "synthetic_corpus"

NER_LABELS = {"NAME", "LOCATION"}


def load_records(split: str):
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] == split:
                yield record


def build_docbin(split: str, nlp) -> DocBin:
    doc_bin = DocBin()
    skipped = 0
    kept = 0

    for record in load_records(split):
        text = record["text"]
        doc = nlp.make_doc(text)
        spans = []

        for ent in record["entities"]:
            if ent["label"] not in NER_LABELS:
                continue  # AADHAAR/PAN — regex's job, not NER's

            span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is None:
                # The character offsets didn't line up with spaCy's
                # tokenization (rare, but must be dropped rather than
                # silently kept — a misaligned span would corrupt
                # training rather than just being missing data).
                skipped += 1
                continue

            spans.append(span)
            kept += 1

        doc.ents = spans
        doc_bin.add(doc)

    print(f"[{split}] kept {kept} entity spans, skipped {skipped} misaligned spans")
    return doc_bin


def main():
    nlp = spacy.blank("en")  # tokenizer only — no pretrained pipeline needed for this step

    train_bin = build_docbin("train", nlp)
    train_bin.to_disk(OUTPUT_DIR / "train.spacy")

    dev_bin = build_docbin("test", nlp)
    dev_bin.to_disk(OUTPUT_DIR / "dev.spacy")

    print(f"\nWrote {OUTPUT_DIR / 'train.spacy'}")
    print(f"Wrote {OUTPUT_DIR / 'dev.spacy'}")


if __name__ == "__main__":
    main()