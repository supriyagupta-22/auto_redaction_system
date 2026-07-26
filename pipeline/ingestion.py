"""
ingestion.py

Extracts plain text from an uploaded file. Currently supports .txt
only — PDF and DOCX support (Roadmap Phase 5) will be added here as
additional branches, without requiring any change to app.py or
anything downstream, since everything past this point only ever
deals with plain text regardless of the original file format.
"""


def extract_text(uploaded_file) -> str:
    """
    `uploaded_file` is a Streamlit UploadedFile object. For .txt files
    this just decodes the raw bytes. PDF/DOCX branches get added here
    in Phase 5 — app.py will never need to know the difference.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    raise ValueError(f"Unsupported file type: {filename}. Only .txt is supported so far.")