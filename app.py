"""
app.py

Streamlit entry point. Ties together ingestion, detection (regex +
NER via the aggregator), masking, and SQLite logging into one working
interface: upload a document, see it redacted side by side with the
original, and download the result.
"""

from collections import Counter

import streamlit as st

from pipeline.aggregator import detect
from pipeline.db_logger import init_db, log_document, log_entities
from pipeline.ingestion import extract_text
from pipeline.masking import mask

st.set_page_config(page_title="Auto Redaction System", layout="wide")

init_db()  # safe to call on every run — a no-op once tables already exist

st.title("Automated Document Redaction System")
st.write("Upload a document to detect and redact PII (Aadhaar, PAN, Names, Locations).")

mode = st.radio(
    "Redaction mode",
    options=["redact", "pseudonymize"],
    format_func=lambda m: "Fixed mask ([REDACTED])" if m == "redact" else "Pseudonymized token",
)

uploaded_file = st.file_uploader("Choose a document", type=["txt"])

if uploaded_file is not None:
    text = extract_text(uploaded_file)
    entities = detect(text)
    redacted_text = mask(text, entities, mode=mode)

    document_id = log_document(uploaded_file.name, uploaded_file.name.split(".")[-1])
    log_entities(document_id, entities, mode=mode)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.text_area("Original document", text, height=400, label_visibility="collapsed")
    with col2:
        st.subheader("Redacted")
        st.text_area("Redacted document", redacted_text, height=400, label_visibility="collapsed")

    st.subheader("Summary")
    counts = Counter(entity.label for entity in entities)
    st.write(f"Detected {len(entities)} entities:")
    for label, count in counts.items():
        st.write(f"- **{label}**: {count}")

    st.download_button(
        label="Download redacted document",
        data=redacted_text,
        file_name=f"redacted_{uploaded_file.name}",
        mime="text/plain",
    )