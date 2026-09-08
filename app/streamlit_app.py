"""Application entry point. Run from the repository root via python -m streamlit."""

import streamlit as st

from app.components.dashboard import render_dashboard
from app.components.documents import render_documents
from app.components.preprocessing import render_preprocessing
from app.components.tokenizer import render_tokenizer
from app.components.trigram import render_trigram
from app.components.workflow import SECTIONS, render_placeholder
from src.config.settings import ConfigurationError, load_settings

st.set_page_config(page_title="Decoder LM · Phase 5", page_icon="📚", layout="wide")
st.sidebar.title("Decoder LM")
st.sidebar.caption("University ML project · Phase 5")
section = st.sidebar.radio("Workspace", ["Dashboard", *SECTIONS])
try:
    settings = load_settings()
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
st.sidebar.caption(f"Profile: {settings.values['environment']}")
if section == "Dashboard":
    render_dashboard(settings)
elif section == "Documents":
    render_documents(settings)
elif section == "Preprocessing":
    render_preprocessing(settings)
elif section == "Tokenizer":
    render_tokenizer(settings)
elif section == "Trigram Model":
    render_trigram(settings)
else:
    render_placeholder(section)
