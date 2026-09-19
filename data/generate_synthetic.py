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
import sys
from pathlib import Path

from faker import Faker

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.verhoeff import generate_check_digit

fake = Faker("en_IN")
random.seed(42)  # reproducible dataset — keep this fixed for now

OUTPUT_DIR = Path(__file__).parent / "synthetic_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Entity generators — one function per PII type
# ---------------------------------------------------------------------

_PAN_HOLDER_TYPES = "ABCFGHLJPT"  # must match _VALID_PAN_HOLDER_TYPES in regex_detectors.py


def generate_pan() -> str:
    """5 letters + 4 digits + 1 letter, e.g. ABCDE1234F. The 4th
    letter is drawn specifically from the valid PAN holder-type codes
    so generated PANs pass the same validation the detector performs."""
    first_three = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    holder_type = random.choice(_PAN_HOLDER_TYPES)
    fifth_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=4))
    last_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{first_three}{holder_type}{fifth_letter}{digits}{last_letter}"


def generate_aadhaar() -> str:
    """12 digits, grouped in 4s: a leading digit from 2-9 (UIDAI never
    issues a leading 0 or 1) followed by 10 more random digits, with a
    real Verhoeff check digit computed and appended as the 12th digit
    — so generated numbers now pass the exact same validation the
    detector performs, instead of failing it almost every time."""
    first_digit = random.choice("23456789")
    middle_digits = "".join(random.choices("0123456789", k=10))
    digits_without_check = first_digit + middle_digits
    check_digit = generate_check_digit(digits_without_check)
    digits = digits_without_check + check_digit
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def generate_name() -> str:
    """Produces a NAME value with varied shape — mostly standard
    2-part 'Firstname Lastname' (matching every real name in the
    original 14 templates), but sometimes 3-part or hyphenated, since
    real-world testing found the model had never seen anything but a
    fixed 2-word shape and struggled with names like 'Siddharth Rao
    Kulkarni' or 'Kavitha-Lakshmi Suresh-Babu'."""
    style = random.choices(
        ["standard", "three_part", "hyphenated"],
        weights=[60, 20, 20],
    )[0]

    if style == "standard":
        return fake.name().strip()
    elif style == "three_part":
        return f"{fake.first_name()} {fake.first_name()} {fake.last_name()}".strip()
    else:  # hyphenated
        return f"{fake.first_name()}-{fake.first_name()} {fake.last_name()}-{fake.last_name()}".strip()


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
#    Templates 15-22 were added after real-world adversarial testing
#    (a long, multi-section memo) revealed the model had never seen:
#    memo/letter headers, title-prefixed names, bullet lists, table-
#    style contact directories, full addresses, sentence-initial
#    pronouns, or organization names — every one of your original 14
#    templates is simple flowing prose with no title prefixes and no
#    structural variety. These templates fill those specific gaps.
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

    """KYC Verification Letter — Account Summary

Dear <<NAME>>,

This letter confirms that your identity has been verified as part of
our Standard Verification Process. Your Aadhaar number is <<AADHAAR>>
and PAN number is <<PAN>>. Kindly visit our <<LOCATION>> branch for
Further Assistance.

Regards,
Compliance Team""",

    """Employment Verification Certificate

To Whom It May Concern,

This is to certify that <<NAME>> is a Permanent Employee based out of
our <<LOCATION>> office, as per our Internal Records. PAN: <<PAN>>,
Aadhaar: <<AADHAAR>>. This Certificate remains valid until further
notice.

Authorized Signatory""",

    """Dear <<NAME>>,

Please find attached the Final Notice regarding your account maintained
at our <<LOCATION>> branch. Kindly submit your PAN (<<PAN>>) and
Aadhaar (<<AADHAAR>>) at the earliest to avoid Service Interruption.

Regards,
Recovery Team""",

    """Subject: Important Update

Dear <<NAME>>,

This is a Courtesy Reminder that your documents are pending submission
at the <<LOCATION>> office. Your Aadhaar (<<AADHAAR>>) and PAN
(<<PAN>>) will be required. Please Note that delays may result in
Account Suspension.

Regards,
Support Team""",

    """Please forward all correspondence regarding <<NAME>> to our
<<LOCATION>> office. PAN: <<PAN>>. Aadhaar: <<AADHAAR>>. Please Note
that response times may vary during Peak Season.""",

    """Our new branch in <<LOCATION>> is now fully operational. For
account queries, please contact <<NAME>>. PAN: <<PAN>>. Aadhaar:
<<AADHAAR>>. The new branch will also handle Customer Service and
Technical Support requests going forward.""",

    # --- New: memo/letter header format (Subject/Prepared by/Date block) ---
    """CONFIDENTIAL MEMO
Subject: Account Verification Update
Prepared by: Compliance Division
Date: 15 September 2026

Dear <<NAME>>,

This memo confirms verification of your registered details. Aadhaar
number: <<AADHAAR>>. PAN: <<PAN>>. Registered office location:
<<LOCATION>>.

Regards,
Compliance Team""",

    # --- New: bullet-list applicant summary ---
    """Applicant Summary:

- Name: <<NAME>>
- Aadhaar: <<AADHAAR>>
- PAN: <<PAN>>
- Branch: <<LOCATION>>
- Note: Please Review before Final Submission

This summary is for internal use only and should not be shared
externally.""",

    # --- New: table-style contact directory, column headers as static text ---
    """Contact Directory

Name                     Phone              Email
--------                 -----              -----
<<NAME>>                 [on file]          [on file]

Please direct all queries to the <<LOCATION>> office. PAN on file:
<<PAN>>. Aadhaar on file: <<AADHAAR>>.""",

    # --- New: full postal address with building/street/pincode ---
    """Their registered address is Flat 12B, Sunview Apartments, MG Road,
<<LOCATION>> — 560038. For verification, contact <<NAME>>. PAN:
<<PAN>>. Aadhaar: <<AADHAAR>>.""",

    # --- New: sentence-initial pronouns (She/His/Her) as non-entities ---
    """She confirmed that <<NAME>> had completed the verification process
successfully. His Aadhaar (<<AADHAAR>>) and PAN (<<PAN>>) were checked
against records held at the <<LOCATION>> branch. Her supervisor
approved the request the same day.""",

    # --- New: title prefix (Dr.) + organization-name distractors ---
    """Dr. <<NAME>> was referred by the Apollo Hospital in <<LOCATION>>
for a routine verification. The referral was reviewed by the Karnataka
Medical Council. PAN on file: <<PAN>>. Aadhaar on file: <<AADHAAR>>.""",

    # --- New: title prefix (Justice) + a different organization distractor ---
    """Justice <<NAME>> presided over the case at the <<LOCATION>> High
Court. The Prime Minister's Office was copied on the correspondence
for informational purposes. Aadhaar: <<AADHAAR>>. PAN: <<PAN>>.""",

    # --- New: dense multi-entity paragraph, closer to real memo density ---
    """Per the audit trail, Mr. <<NAME>> made a transfer from the
<<LOCATION>> branch office. His Aadhaar (<<AADHAAR>>) and PAN
(<<PAN>>) were verified by the compliance officer. A duplicate alert
was sent to the regional office, and the Reserve Bank of India
guidelines were followed throughout the process.""",

    # --- New: comma-separated adjacent locations (Area, City pattern) ---
    # Real-world testing found only the FIRST of two comma-separated
    # locations getting tagged ("Koramangala, Bengaluru" -> only
    # Koramangala caught) — no prior template ever had two LOCATION
    # tokens in immediate sequence, only ever one per template.
    """Registered Address: <<LOCATION>>, <<LOCATION>>. For further
verification, please contact <<NAME>>. PAN on file: <<PAN>>. Aadhaar
on file: <<AADHAAR>>.""",

    """<<NAME>> can be reached at our office located in <<LOCATION>>,
<<LOCATION>>. PAN: <<PAN>>. Aadhaar: <<AADHAAR>>.""",
]


# ---------------------------------------------------------------------
# 3. Fill a template and record ground-truth entity offsets
# ---------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"<<(\w+)>>")

# A broad pool of generic, two/three-word capitalized business phrases
# that are deliberately NOT entities. 1-2 of these get appended (as
# full sentences) to EVERY generated document, giving the NER model
# far broader and more randomized exposure to "capitalized phrase that
# isn't a name or location" than a handful of hand-picked phrases in
# a couple of templates ever could — the goal is teaching the general
# pattern, not memorizing a growing list of specific exceptions.
#
# Note: "Service Agreement" / "Escalation Matrix" are deliberately
# excluded from this pool — they're used in test_ner_generalization.py
# as held-out phrases to check whether this actually generalizes.
DISTRACTOR_PHRASES = [
    "Final Notice", "Rental Agreement", "Terms and Conditions", "Account Summary",
    "Important Update", "Please Note", "For Your Information", "Standard Procedure",
    "Internal Records", "Further Assistance", "Permanent Employee", "Courtesy Reminder",
    "Account Suspension", "Important Notice", "Reference Number", "Application Form",
    "Processing Fee", "Customer Service", "Technical Support", "Quality Assurance",
    "Human Resources", "General Manager", "Senior Officer", "Branch Manager",
    "Head Office", "Registered Address", "Contact Information", "Payment Received",
    "Outstanding Balance", "Due Date", "Grace Period", "Late Fee",
    "Subject To Change", "Kindly Note", "Please Ensure", "Immediate Action",
    "Prompt Response", "Necessary Action", "Required Documents", "Supporting Documents",
    "Valid Proof", "Original Copy", "Self Attested", "Duly Signed",
    "Complete Application", "Pending Approval", "Under Review", "Further Details",
    "Additional Information", "Terms Apply",
    # Single generic tech/business nouns confirmed via diagnostic testing
    # to be wrongly tagged LOCATION, likely due to pretrained bias
    # inherited from en_core_web_sm rather than anything in our own
    # training data — added directly as distractors once confirmed.
    "Matrix", "Portal", "Dashboard", "System", "Module", "Gateway", "Protocol",
    "Platform", "Interface", "Console", "Network", "Database", "Server",
    "Pipeline", "Workflow",
    # Confirmed failing in round 2 diagnostic — previously a clean control
    # group, now genuinely broken after the round-1 fix shifted the
    # decision boundary.
    "Framework", "Directory", "Registry",
    # Hardcoded directly rather than left to the garbled-noise generator's
    # random corruption sampling — "Aacrawr" is only one of several
    # possible outputs of corrupting "Aadhaar", so it wasn't guaranteed
    # to actually appear in any given generated corpus. This guarantees
    # the exact real-world observed string gets covered every run.
    "Aacrawr",
]

DISTRACTOR_SENTENCE_FRAMES = [
    "Note that {phrase} may apply in this case.",
    "This is subject to {phrase} as per company policy.",
    "Kindly refer to the {phrase} for more details.",
    "{phrase} is required before this can be processed further.",
    "As per our {phrase}, this matter will be reviewed shortly.",
    "The {phrase} outlined here remains in effect.",
    "Refer to the {phrase} section for further clarification.",
]


def add_distractor_sentences(text: str, n: int = 2) -> str:
    """Appends n randomly chosen distractor sentences to the END of a
    document. Always appended after all existing content — never
    inserted earlier — so none of the already-computed entity offsets
    are disturbed."""
    sentences = []
    for _ in range(n):
        phrase = random.choice(DISTRACTOR_PHRASES)
        frame = random.choice(DISTRACTOR_SENTENCE_FRAMES)
        sentences.append(frame.format(phrase=phrase))
    return text + "\n\n" + " ".join(sentences)


# --- OCR-noise distractors -------------------------------------------
# A different problem from the phrase distractors above: real OCR
# occasionally misreads a genuine word into something nonsense-shaped
# but still capitalized and word-like ("Aadhaar" -> "Aacrawr" was an
# actual observed case). This teaches the NER model that garbled,
# meaningless capitalized fragments aren't automatically entities
# either — a different failure mode than "real phrase mistaken for a
# name," so it needs its own kind of negative example.

OCR_CONFUSION_PAIRS = {
    "O": "0", "0": "O",
    "I": "1", "1": "I", "l": "1",
    "D": "O", "B": "8", "8": "B",
    "S": "5", "5": "S",
}

OCR_NOISE_BASE_WORDS = [
    "Aadhaar", "Verification", "Number", "Account", "Reference",
    "Certificate", "Document", "Application", "Branch", "Office",
    "Employee", "Customer", "Signature", "Confirmation",
    "Registration", "Identification", "Processing", "Submission",
]


def _corrupt_word(word: str) -> str:
    """Applies one random OCR-realistic corruption to a word: a
    character-confusion substitution, or a dropped/duplicated
    character — the same kinds of errors Tesseract actually produces,
    not purely random noise."""
    chars = list(word)
    corruption = random.choice(["substitute", "drop", "duplicate"])

    if corruption == "substitute":
        candidates = [i for i, c in enumerate(chars) if c.upper() in OCR_CONFUSION_PAIRS]
        if candidates:
            idx = random.choice(candidates)
            chars[idx] = OCR_CONFUSION_PAIRS[chars[idx].upper()]
    elif corruption == "drop" and len(chars) > 3:
        del chars[random.randrange(len(chars))]
    elif corruption == "duplicate":
        idx = random.randrange(len(chars))
        chars.insert(idx, chars[idx])

    return "".join(chars)


def add_garbled_noise(text: str, n: int = 1) -> str:
    """Appends n short OCR-garbled word fragments to the END of a
    document — e.g. 'Aacrawr Nurnber.' Always appended after all
    existing content, so entity offsets are never disturbed."""
    fragments = []
    for _ in range(n):
        n_words = random.randint(1, 2)
        words = [_corrupt_word(random.choice(OCR_NOISE_BASE_WORDS)) for _ in range(n_words)]
        fragments.append(" ".join(words))
    return text + " " + " ".join(fragments) + "."


# --- Organization-name distractors ------------------------------------
# Real-world adversarial testing found institutional names — several of
# which literally embed a real place name ("Karnataka High Court",
# "Prime Minister's Office") — getting wrongly tagged as LOCATION or
# NAME. ORG was never a category this system was designed to recognize
# at all, but the model still needs explicit exposure to these phrases
# as non-entities, especially ones containing a location-like word.

ORG_NAME_DISTRACTORS = [
    "Karnataka High Court", "State Bank of India", "Prime Minister's Office",
    "Apollo Hospital", "Reserve Bank of India", "Income Tax Department",
    "Supreme Court of India", "Indian Institute of Technology",
    "Ministry of Finance", "Election Commission of India",
    "Karnataka Medical Council", "Tamil Nadu Housing Board",
]

ORG_SENTENCE_FRAMES = [
    "This matter was reviewed by the {org}.",
    "A copy was forwarded to the {org} for records.",
    "The {org} confirmed receipt of the documents.",
    "Further correspondence should be directed through the {org}.",
]


def add_org_distractor(text: str, n: int = 1) -> str:
    """Appends n sentences mentioning a realistic Indian organization
    name as background, non-entity context."""
    sentences = []
    for _ in range(n):
        org = random.choice(ORG_NAME_DISTRACTORS)
        frame = random.choice(ORG_SENTENCE_FRAMES)
        sentences.append(frame.format(org=org))
    return text + " " + " ".join(sentences)


# --- Sentence-initial pronoun distractors ------------------------------
# Real-world testing found "She" and "He" at the start of a sentence
# occasionally swept into a LOCATION span — plausibly an echo of the
# "capitalized word in a familiar entity-adjacent position" shortcut
# resurfacing on unfamiliar sentence shapes. This gives direct,
# repeated exposure to common pronouns as sentence openers.

PRONOUN_SENTENCE_FRAMES = [
    "She confirmed the details were accurate.",
    "He verified the submission the same day.",
    "Her office processed the request promptly.",
    "His department reviewed the file without delay.",
]


def add_pronoun_distractor(text: str) -> str:
    """Appends one sentence starting with a common pronoun."""
    return text + " " + random.choice(PRONOUN_SENTENCE_FRAMES)


# --- Company signature-block distractors -------------------------------
# Real-world testing found a letter's OWN sender company name, in its
# closing signature block, wrongly tagged LOCATION ("Horizon Financial
# Services Pvt. Ltd." in a "Regards, Team, Company" sign-off). Every
# existing template's own baked-in ending is a bare "Regards, Team"
# with no company name at all, so this exact pattern — company name in
# a signature block — was never modeled anywhere.

COMPANY_NAME_DISTRACTORS = [
    "Horizon Financial Services Pvt. Ltd.",
    "Meridian Business Solutions LLP",
    "Sunrise Capital Advisors Pvt. Ltd.",
    "Bluewave Consulting Services Pvt. Ltd.",
    "Crestline Technologies Pvt. Ltd.",
    "Orion Financial Group Pvt. Ltd.",
    "Silverline Enterprises LLP",
    "Northgate Solutions Pvt. Ltd.",
]

SIGNATURE_TEAM_NAMES = ["Accounts Team", "Compliance Team", "Support Team", "Operations Desk", "Client Services"]

SIGNATURE_FRAMES = [
    "Regards,\n{team}\n{company}",
    "Sincerely,\n{team}\n{company}",
    "Thank you,\n{team}\n{company}",
]


def add_company_signature(text: str) -> str:
    """Appends a realistic closing signature block naming the sender's
    own company — teaches the model that an organization name in this
    position is not a LOCATION or NAME entity."""
    team = random.choice(SIGNATURE_TEAM_NAMES)
    company = random.choice(COMPANY_NAME_DISTRACTORS)
    frame = random.choice(SIGNATURE_FRAMES)
    return text + "\n\n" + frame.format(team=team, company=company)


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
            text = add_distractor_sentences(text, n=random.randint(1, 2))
            text = add_garbled_noise(text, n=1)
            text = add_org_distractor(text, n=1)
            text = add_pronoun_distractor(text)
            text = add_company_signature(text)

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
    generate_corpus(n_documents=1200)