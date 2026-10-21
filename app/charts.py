"""Member 3 - chart helpers (thin wrappers around src.reporting)."""
import streamlit as st

from src import reporting


def show(fig):
    st.pyplot(fig)


confusion = reporting.confusion_fig
comparison = reporting.comparison_fig
roc = reporting.roc_fig
importance = reporting.importance_fig
score_hist = reporting.score_hist_fig
