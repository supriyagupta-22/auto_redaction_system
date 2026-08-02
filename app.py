"""
app.py

Streamlit entry point. Accepts one or more documents (.txt, .pdf,
.docx), detects PII, lets the user choose which categories to
actually redact and in which mode, and logs every run to SQLite.
"""

from collections import Counter

import streamlit as st

from pipeline.aggregator import detect, filter_by_category
from pipeline.db_logger import init_db, log_document, log_entities
from pipeline.ingestion import extract_text
from pipeline.masking import mask

st.set_page_config(page_title="Auto Redaction System", layout="wide")

init_db()

st.title("Automated Document Redaction System")
st.write("Upload one or more documents to detect and redact PII (Aadhaar, PAN, Names, Locations).")

with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Redaction mode",
        options=["redact", "pseudonymize"],
        format_func=lambda m: "Fixed mask ([REDACTED])" if m == "redact" else "Pseudonymized token",
    )

    st.subheader("Government IDs")
    redact_aadhaar = st.checkbox("Aadhaar", value=True)
    redact_pan = st.checkbox("PAN", value=True)
    redact_passport = st.checkbox("Passport", value=True)

    st.subheader("Financial")
    redact_card = st.checkbox("Credit/Debit Card", value=True)
    redact_gstin = st.checkbox("GSTIN", value=True)
    redact_ifsc = st.checkbox("IFSC Code", value=True)

    st.subheader("Contact Info")
    redact_email = st.checkbox("Email Address", value=True)
    redact_phone = st.checkbox("Phone Number", value=True)

    st.subheader("AI-Detected")
    redact_name = st.checkbox("Names", value=True)
    redact_location = st.checkbox("Locations", value=True)

    enabled_categories = set()
    if redact_aadhaar:
        enabled_categories.add("AADHAAR")
    if redact_pan:
        enabled_categories.add("PAN")
    if redact_passport:
        enabled_categories.add("PASSPORT")
    if redact_card:
        enabled_categories.add("CARD")
    if redact_gstin:
        enabled_categories.add("GSTIN")
    if redact_ifsc:
        enabled_categories.add("IFSC")
    if redact_email:
        enabled_categories.add("EMAIL")
    if redact_phone:
        enabled_categories.add("PHONE")
    if redact_name:
        enabled_categories.add("NAME")
    if redact_location:
        enabled_categories.add("LOCATION")

uploaded_files = st.file_uploader(
    "Choose one or more documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    if not enabled_categories:
        st.warning("No redaction categories are selected in the sidebar — documents will be shown unredacted.")

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

        all_entities = detect(text)
        entities = filter_by_category(all_entities, enabled_categories)
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
        st.write(f"Redacted {len(entities)} entities (selected categories only):")
        for label, count in counts.items():
            st.write(f"- **{label}**: {count}")

        skipped = len(all_entities) - len(entities)
        if skipped > 0:
            st.caption(
                f"{skipped} additional entit{'y was' if skipped == 1 else 'ies were'} "
                f"detected but left unredacted because their category is disabled in the sidebar."
            )

        st.download_button(
            label="Download redacted document",
            data=redacted_text,
            file_name=f"redacted_{uploaded_file.name}.txt",
            mime="text/plain",
            key=f"download_{i}_{uploaded_file.name}",
        )