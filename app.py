"""
app.py

Streamlit entry point. Accepts one or more documents (.txt, .pdf,
.docx, or images for OCR), detects PII, lets the user choose which
categories to redact and in which mode, and logs every run to SQLite.
"""

from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from pipeline.aggregator import detect, filter_by_category
from pipeline.db_logger import (
    export_csv,
    get_all_time_stats,
    init_db,
    log_document,
    log_entities,
    severity_counts,
)
from pipeline.ingestion import extract_text, is_low_confidence_extraction
from pipeline.masking import mask

st.set_page_config(
    page_title="Auto Redaction System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ---------------------------------------------------------------------
# Category metadata — icon, color, and display order. Used throughout
# the sidebar, entity badges, and summaries so every category looks
# consistent wherever it appears.
# ---------------------------------------------------------------------

CATEGORY_META = {
    "AADHAAR":  {"icon": "🆔", "color": "#DC2626", "group": "Government IDs"},
    "PAN":      {"icon": "🧾", "color": "#DC2626", "group": "Government IDs"},
    "PASSPORT": {"icon": "🛂", "color": "#DC2626", "group": "Government IDs"},
    "CARD":     {"icon": "💳", "color": "#DC2626", "group": "Financial"},
    "GSTIN":    {"icon": "🏢", "color": "#EA580C", "group": "Financial"},
    "IFSC":     {"icon": "🏦", "color": "#2563EB", "group": "Financial"},
    "EMAIL":    {"icon": "📧", "color": "#EA580C", "group": "Contact Info"},
    "PHONE":    {"icon": "📱", "color": "#EA580C", "group": "Contact Info"},
    "NAME":     {"icon": "👤", "color": "#EA580C", "group": "AI-Detected"},
    "LOCATION": {"icon": "📍", "color": "#16A34A", "group": "AI-Detected"},
}
GROUP_ORDER = ["Government IDs", "Financial", "Contact Info", "AI-Detected"]

SEVERITY_COLORS = {"CRITICAL": "#DC2626", "HIGH": "#EA580C", "LOW": "#16A34A"}


# ---------------------------------------------------------------------
# Custom styling — kept minimal and additive; Streamlit's own theme
# (.streamlit/config.toml) handles the base palette, this just adds
# badges, card polish, and a header banner CSS can't otherwise do.
# ---------------------------------------------------------------------

st.markdown("""
<style>
    .app-header {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; }
    .app-header p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .entity-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 6px 3px 0;
    }

    .metric-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>🔒 Automated Document Redaction System</h1>
    <p>Detect and redact Aadhaar, PAN, Passport, Card, GSTIN, IFSC, Email, Phone,
    Names, and Locations — locally, with no data leaving your machine.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    mode = st.radio(
        "Redaction mode",
        options=["redact", "pseudonymize"],
        format_func=lambda m: "🔲 Fixed mask ([REDACTED])" if m == "redact" else "🔑 Pseudonymized token",
    )

    st.markdown("### 🎯 Categories to Redact")

    for label in CATEGORY_META:
        if f"chk_{label}" not in st.session_state:
            st.session_state[f"chk_{label}"] = True

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Select all", use_container_width=True):
            for label in CATEGORY_META:
                st.session_state[f"chk_{label}"] = True
    with col_b:
        if st.button("Clear all", use_container_width=True):
            for label in CATEGORY_META:
                st.session_state[f"chk_{label}"] = False

    enabled_categories = set()
    for group in GROUP_ORDER:
        with st.expander(group, expanded=True):
            for label, meta in CATEGORY_META.items():
                if meta["group"] != group:
                    continue
                # Note: only `key=` is passed here, never `value=` —
                # combining both on a widget that persists across
                # reruns causes Streamlit to silently ignore `value=`
                # after the first render, which would make the Select
                # all/Clear all buttons above appear to do nothing.
                checked = st.checkbox(f"{meta['icon']} {label.title()}", key=f"chk_{label}")
                if checked:
                    enabled_categories.add(label)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def render_entity_badges(entities):
    counts = Counter(e.label for e in entities)
    html = ""
    for label, count in sorted(counts.items()):
        meta = CATEGORY_META.get(label, {"icon": "🔹", "color": "#64748B"})
        html += (
            f'<span class="entity-badge" style="background:{meta["color"]}18;'
            f'color:{meta["color"]};border:1px solid {meta["color"]}40;">'
            f'{meta["icon"]} {label} × {count}</span>'
        )
    st.markdown(html, unsafe_allow_html=True)


def build_severity_chart(counts: dict) -> go.Figure:
    order = ["CRITICAL", "HIGH", "LOW"]
    fig = go.Figure(data=[go.Bar(
        x=order,
        y=[counts.get(s, 0) for s in order],
        marker_color=[SEVERITY_COLORS[s] for s in order],
    )])
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Entities",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------
# Main area — upload and results
# ---------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "📁 Choose one or more documents",
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.container(border=True).markdown("""
**How it works**

1. Upload one or more documents — plain text, PDF, Word, or scanned images
2. Choose which PII categories to redact in the sidebar
3. Pick fixed masking or reversible pseudonymized tokens
4. Review the redacted output, risk summary, and download an audit-ready CSV report

Every detection runs locally — nothing is sent to any external service.
""")

if uploaded_files:
    if not enabled_categories:
        st.warning("No redaction categories are selected in the sidebar — documents will be shown unredacted.")

    tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files]) if len(uploaded_files) > 1 else [st.container()]

    for i, (uploaded_file, tab) in enumerate(zip(uploaded_files, tabs)):
        with tab:
            if len(uploaded_files) == 1:
                st.markdown(f"#### 📄 {uploaded_file.name}")

            try:
                text = extract_text(uploaded_file)
            except ValueError as e:
                st.error(str(e))
                continue

            if not text.strip():
                st.warning("No extractable text found in this file — it may be a scanned/image-only document.")
                continue

            if is_low_confidence_extraction(text, uploaded_file.name):
                st.warning(
                    "⚠️ This looks like a low-quality scan — very little text was extracted. "
                    "Detected entities below may be incomplete. Consider re-scanning at higher "
                    "resolution or with better lighting/focus."
                )

            all_entities = detect(text)
            entities = filter_by_category(all_entities, enabled_categories)
            redacted_text = mask(text, entities, mode=mode)

            document_id = log_document(uploaded_file.name, uploaded_file.name.split(".")[-1])
            log_entities(document_id, entities, mode=mode)

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**📝 Original**")
                    st.text_area(
                        "Original document", text, height=280,
                        label_visibility="collapsed", key=f"orig_{i}_{uploaded_file.name}",
                    )
            with col2:
                with st.container(border=True):
                    st.markdown("**🔒 Redacted**")
                    st.text_area(
                        "Redacted document", redacted_text, height=280,
                        label_visibility="collapsed", key=f"redacted_{i}_{uploaded_file.name}",
                    )

            with st.container(border=True):
                st.markdown(f"**✅ Redacted {len(entities)} entities**")
                if entities:
                    render_entity_badges(entities)
                skipped = len(all_entities) - len(entities)
                if skipped > 0:
                    st.caption(
                        f"ℹ️ {skipped} additional entit{'y was' if skipped == 1 else 'ies were'} "
                        f"detected but left unredacted because their category is disabled in the sidebar."
                    )

            risk_col, actions_col = st.columns([2, 1])
            with risk_col:
                with st.container(border=True):
                    st.markdown("**📊 Risk Summary**")
                    risk_counts = severity_counts(entities)
                    chart_c, metric_c = st.columns([2, 1])
                    with chart_c:
                        st.plotly_chart(
                            build_severity_chart(risk_counts), use_container_width=True,
                            key=f"chart_{i}_{uploaded_file.name}",
                        )
                    with metric_c:
                        st.metric("🔴 Critical", risk_counts["CRITICAL"])
                        st.metric("🟠 High", risk_counts["HIGH"])
                        st.metric("🟢 Low", risk_counts["LOW"])
            with actions_col:
                with st.container(border=True):
                    st.markdown("**⬇️ Downloads**")
                    st.download_button(
                        label="Redacted document",
                        data=redacted_text,
                        file_name=f"redacted_{uploaded_file.name}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"download_{i}_{uploaded_file.name}",
                    )
                    st.download_button(
                        label="Audit report (CSV)",
                        data=export_csv(document_id),
                        file_name=f"audit_{uploaded_file.name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"csv_{i}_{uploaded_file.name}",
                    )


# ---------------------------------------------------------------------
# All-time dashboard
# ---------------------------------------------------------------------

st.divider()
with st.expander("📈 All-time dashboard (across every document ever processed)"):
    stats = get_all_time_stats()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Documents processed", stats["total_documents"])
    m2.metric("🔎 Entities redacted", stats["total_events"])
    m3.metric("🔴 Critical", stats["by_severity"].get("CRITICAL", 0))
    m4.metric("🟢 Low", stats["by_severity"].get("LOW", 0))

    st.markdown("**By category**")
    cat_html = ""
    for category, count in sorted(stats["by_category"].items()):
        meta = CATEGORY_META.get(category, {"icon": "🔹", "color": "#64748B"})
        cat_html += (
            f'<span class="entity-badge" style="background:{meta["color"]}18;'
            f'color:{meta["color"]};border:1px solid {meta["color"]}40;">'
            f'{meta["icon"]} {category} × {count}</span>'
        )
    st.markdown(cat_html, unsafe_allow_html=True)