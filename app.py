"""
app.py

Streamlit entry point. Accepts one or more documents (.txt, .pdf,
.docx), detects and redacts PII in each, and logs every run to SQLite.
"""

from collections import Counter

import streamlit as st

from pipeline.aggregator import detect
from pipeline.db_logger import init_db, log_document, log_entities
from pipeline.ingestion import extract_text
from pipeline.masking import mask

st.set_page_config(page_title="Auto Redaction System", layout="wide")

init_db()

st.title("Automated Document Redaction System")
st.write("Upload one or more documents to detect and redact PII (Aadhaar, PAN, Names, Locations).")

mode = st.radio(
    "Redaction mode",
    options=["redact", "pseudonymize"],
    format_func=lambda m: "Fixed mask ([REDACTED])" if m == "redact" else "Pseudonymized token",
)

uploaded_files = st.file_uploader(
    "Choose one or more documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.markdown(f"### {uploaded_file.name}")

        try:
            text = extract_text(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            continue

        if not text.strip():
            st.warning("No extractable text found in this file — it may be a scanned/image-only document.")
            continue

        entities = detect(text)
        redacted_text = mask(text, entities, mode=mode)

        document_id = log_document(uploaded_file.name, uploaded_file.name.split(".")[-1])
        log_entities(document_id, entities, mode=mode)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.text_area(
                "Original document", text, height=300,
                label_visibility="collapsed", key=f"orig_{i}_{uploaded_file.name}",
            )
        with col2:
            st.subheader("Redacted")
            st.text_area(
                "Redacted document", redacted_text, height=300,
                label_visibility="collapsed", key=f"redacted_{i}_{uploaded_file.name}",
            )

        st.subheader("Summary")
        counts = Counter(entity.label for entity in entities)
        st.write(f"Detected {len(entities)} entities:")
        for label, count in counts.items():
            st.write(f"- **{label}**: {count}")

        st.download_button(
            label="Download redacted document",
            data=redacted_text,
            file_name=f"redacted_{uploaded_file.name}.txt",
            mime="text/plain",
            key=f"download_{i}_{uploaded_file.name}",
        )