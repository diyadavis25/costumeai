"""Outfit AI - Streamlit entry point.   Run from the project root:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Outfit AI", page_icon="👗", layout="wide")

from app import dashboard, login, ui  # noqa: E402

ui.inject_css()
if not login.require_login():
    st.stop()
dashboard.init_state()

with st.sidebar:
    ui.html('<div class="brand">Outfit AI</div><div style="opacity:.7;font-size:.85rem;margin-bottom:14px">Smart mirror for outfit compatibility</div>')
    page = st.radio("Navigate", dashboard.PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    login.logout_button()
    st.divider()
    o = st.session_state.outfit
    if all(o.values()):
        st.caption("Current outfit: " + ", ".join(f"{o[s]['color']} {s}" for s in ("top", "bottom", "shoes")))
    else:
        st.caption("No outfit analysed yet.")

{
    "Home": dashboard.page_home,
    "Analyze Outfit": dashboard.page_analyze,
    "AI Analysis": dashboard.page_analysis,
    "Recommendations": dashboard.page_recommendations,
    "My Wardrobe": dashboard.page_wardrobe,
    "Model Performance": dashboard.page_performance,
    "About": dashboard.page_about,
}[page]()