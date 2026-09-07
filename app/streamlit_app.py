"""Phase 1 application entry point. Run from the repository root via python -m streamlit."""

import streamlit as st

from app.components.dashboard import render_dashboard
from app.components.workflow import SECTIONS, render_placeholder
from src.config.settings import ConfigurationError, load_settings

st.set_page_config(page_title="Decoder LM · Phase 1", page_icon="📚", layout="wide")
st.sidebar.title("Decoder LM")
st.sidebar.caption("University ML project · Phase 1")
section = st.sidebar.radio("Workspace", ["Dashboard", *SECTIONS])
try:
    settings = load_settings()
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
st.sidebar.caption(f"Profile: {settings.values['environment']}")
if section == "Dashboard":
    render_dashboard(settings)
else:
    render_placeholder(section)
