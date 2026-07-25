"""
generate_synthetic.py

Generates a synthetic, labeled corpus of business-correspondence-style
documents with embedded Indian PII (names, Aadhaar numbers, PAN numbers,
locations) for training and evaluating the redaction pipeline.

Output:
    synthetic_corpus/doc_0001.txt, doc_0002.txt, ...
    synthetic_corpus/labels.jsonl   (one JSON record per document, with
        the exact character offsets and type of every inserted PII
        value, plus a train/test split assignment)

Run:
    python generate_synthetic.py
"""

import json
import random
import re
from pathlib import Path

from faker import Faker

fake = Faker("en_IN")
random.seed(42)  # reproducible dataset — keep this fixed for now

OUTPUT_DIR = Path(__file__).parent / "synthetic_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Entity generators — one function per PII type
# ---------------------------------------------------------------------

def generate_pan() -> str:
    """5 letters + 4 digits + 1 letter, e.g. ABCDE1234F"""
    letters1 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    digits = "".join(random.choices("0123456789", k=4))
    letter2 = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{letters1}{digits}{letter2}"


def generate_aadhaar() -> str:
    """12 digits, grouped in 4s. Avoids a leading 0/1, since UIDAI never
    issues those — keeps the synthetic data realistic."""
    first_digit = random.choice("23456789")
    rest = "".join(random.choices("0123456789", k=11))
    digits = first_digit + rest
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def generate_name() -> str:
    return fake.name().strip()


def generate_location() -> str:
    return fake.city().strip()


ENTITY_GENERATORS = {
    "PAN": generate_pan,
    "AADHAAR": generate_aadhaar,
    "NAME": generate_name,
    "LOCATION": generate_location,
}


# ---------------------------------------------------------------------
# 2. Document templates
#    Placeholders use <<TOKEN>> syntax matching ENTITY_GENERATORS keys.
#    Add more templates over time as you go — variety here is what
#    actually improves how well the NER model generalizes later.
# ---------------------------------------------------------------------

TEMPLATES = [
    """Dear <<NAME>>,

This letter is to confirm that your account has been successfully
verified. Your PAN on file is <<PAN>> and your Aadhaar number in our
records is <<AADHAAR>>. Please visit our branch in <<LOCATION>> if you
have any questions regarding this verification.

Regards,
Accounts Team""",

    """To Whom It May Concern,

This is to certify that <<NAME>>, residing in <<LOCATION>>, has been
employed with our organization since 2021. For verification purposes,
the employee's PAN is <<PAN>> and Aadhaar number is <<AADHAAR>>.

Sincerely,
HR Department""",

    """Subject: KYC Update Required

Dear <<NAME>>,

Our records indicate that your KYC details require an update. Kindly
submit a copy of your PAN card (<<PAN>>) and Aadhaar card (<<AADHAAR>>)
at your earliest convenience to our <<LOCATION>> office.

Thank you,
Compliance Desk""",

    """Loan Application Reference

Applicant Name: <<NAME>>
Aadhaar Number: <<AADHAAR>>
PAN Number: <<PAN>>
Branch: <<LOCATION>>

This application has been received and is under review. You will be
notified of the outcome within 7 working days.""",

    """The <<LOCATION>> office will remain closed on Monday and Tuesday
for maintenance. Please redirect all queries to <<NAME>> until further
notice. Aadhaar <<AADHAAR>> and PAN <<PAN>> have already been verified.""",

    """<<NAME>> submitted the required paperwork on Friday. The case has
been forwarded to our <<LOCATION>> office for final processing. PAN on
file: <<PAN>>. Aadhaar on file: <<AADHAAR>>.""",

    """Effective next Wednesday, all correspondence regarding
<<NAME>>'s application (PAN <<PAN>>, Aadhaar <<AADHAAR>>) should be
directed to the <<LOCATION>> regional office.""",

    """We regret to inform you that the branch in <<LOCATION>> will be
shut on Saturday and Sunday. Contact <<NAME>> for urgent matters. For
reference, PAN <<PAN>> and Aadhaar <<AADHAAR>> are on record.""",
]


# ---------------------------------------------------------------------
# 3. Fill a template and record ground-truth entity offsets
# ---------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"<<(\w+)>>")


def fill_template(template: str):
    """
    Replaces every <<TOKEN>> in the template with a generated value,
    building the output text piece by piece so we can record the exact
    (start, end) character offset of every inserted entity — this is
    what makes the dataset usable for NER training and evaluation.
    """
    output = []
    entities = []
    cursor = 0

    for match in TOKEN_PATTERN.finditer(template):
        output.append(template[cursor:match.start()])  # static text before token

        token = match.group(1)
        value = ENTITY_GENERATORS[token]()

        start_offset = sum(len(chunk) for chunk in output)
        output.append(value)
        end_offset = sum(len(chunk) for chunk in output)

        entities.append({"start": start_offset, "end": end_offset, "label": token})
        cursor = match.end()

    output.append(template[cursor:])  # trailing static text
    return "".join(output), entities


# ---------------------------------------------------------------------
# 4. Generate the full corpus, with a held-out test split reserved now
#    (this is the test set Roadmap Phase 9 needs — building it here
#    means you won't be scrambling for labeled data later)
# ---------------------------------------------------------------------

def generate_corpus(n_documents: int = 250, test_ratio: float = 0.2):
    n_test = int(n_documents * test_ratio)
    splits = ["test"] * n_test + ["train"] * (n_documents - n_test)
    random.shuffle(splits)

    labels_path = OUTPUT_DIR / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as labels_file:
        for i in range(1, n_documents + 1):
            template = random.choice(TEMPLATES)
            text, entities = fill_template(template)

            doc_filename = f"doc_{i:04d}.txt"
            (OUTPUT_DIR / doc_filename).write_text(text, encoding="utf-8")

            record = {
                "filename": doc_filename,
                "text": text,
                "entities": entities,
                "split": splits[i - 1],
            }
            labels_file.write(json.dumps(record) + "\n")

    print(f"Generated {n_documents} documents ({n_documents - n_test} train / {n_test} test) in {OUTPUT_DIR}")
    print(f"Ground-truth labels written to {labels_path}")


if __name__ == "__main__":
    generate_corpus(n_documents=400)