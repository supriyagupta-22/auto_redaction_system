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

st.set_page_config(
    page_title="Auto Redaction System", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional and elegant look
st.markdown("""
<style>
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #e74c3c;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #7f8c8d;
    }
    
    /* Text areas */
    .stTextArea textarea {
        border-radius: 8px;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stDownloadButton>button {
        background-color: #27ae60 !important;
        color: white !important;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stDownloadButton>button:hover {
        background-color: #2ecc71 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

init_db()  # safe to call on every run — a no-op once tables already exist

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("Configure the redaction engine and upload your document below.")
    
    st.markdown("---")
    
    mode = st.radio(
        "Redaction Mode",
        options=["redact", "pseudonymize"],
        format_func=lambda m: "Fixed Mask ([REDACTED])" if m == "redact" else "Pseudonymized Token",
        help="Choose how the detected PII should be masked."
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Document", type=["txt"], help="Upload a plain text file for redaction.")

# --- MAIN CONTENT ---
st.title("🛡️ Auto Redaction System")
st.markdown("Automatically detect and redact sensitive Personally Identifiable Information (PII) such as Aadhaar, PAN, Names, and Locations from your documents.")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        entities = detect(text)
        redacted_text = mask(text, entities, mode=mode)
        
        # Log to DB
        document_id = log_document(uploaded_file.name, uploaded_file.name.split(".")[-1])
        log_entities(document_id, entities, mode=mode)
        
    st.success("Document processed successfully!", icon="✅")
    
    # --- METRICS ---
    st.markdown("### 📊 Detection Summary")
    counts = Counter(entity.label for entity in entities)
    
    # Create columns dynamically based on detected entity types + total
    metric_cols = st.columns(len(counts) + 1 if counts else 1)
    
    with metric_cols[0]:
        st.metric("Total Entities", len(entities))
        
    for i, (label, count) in enumerate(counts.items(), start=1):
        with metric_cols[i]:
            st.metric(label, count)
            
    st.markdown("---")
    
    # --- DOCUMENT COMPARISON ---
    st.markdown("### 📄 Document Viewer")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Document**")
        st.text_area("Original", text, height=500, label_visibility="collapsed", disabled=True)
    with col2:
        st.markdown("**Redacted Document**")
        st.text_area("Redacted", redacted_text, height=500, label_visibility="collapsed", disabled=True)
        
    st.markdown("---")
    
    # --- DOWNLOAD & DETAILS ---
    action_col1, action_col2 = st.columns([1, 2])
    with action_col1:
        st.download_button(
            label="⬇️ Download Redacted Document",
            data=redacted_text,
            file_name=f"redacted_{uploaded_file.name}",
            mime="text/plain",
            use_container_width=True
        )
    with action_col2:
        with st.expander("View Detailed Entity Logs"):
            if entities:
                log_data = [{"Entity Type": e.label, "Value": e.text, "Start": e.start, "End": e.end} for e in entities]
                st.dataframe(log_data, use_container_width=True)
            else:
                st.info("No entities detected in this document.")
else:
    # Empty state when no file is uploaded
    st.info("👈 Please upload a document from the sidebar to begin.")
    
    # Optional: Display some features or instructions
    st.markdown("""
    ### System Features
    - **Advanced NER**: Uses state-of-the-art NLP to detect names and locations.
    - **Regex Matching**: Precisely identifies structured data like Aadhaar and PAN cards.
    - **Multiple Modes**: Choose between standard redaction (`[REDACTED]`) and pseudonymization.
    - **Secure & Private**: Processing happens locally, and audit logs are stored securely.
    """)