"""Occasion check panel, shown below the analysis on the Analyze Outfit and AI Analysis pages."""
import streamlit as st

from app import ui
from src import occasion
from src.config import SLOTS

ICON = {"good": "✅", "warn": "⚠️", "bad": "❌"}


def panel():
    o = st.session_state.get("outfit")
    if not o or not all(o.get(s) for s in SLOTS):
        return
    from app import dashboard
    res = dashboard.current_result()
    items = {s: o[s] for s in SLOTS}

    st.divider()
    ui.section("Occasion check", "Where are you wearing this? Get advice for the setting.")
    detected, why = occasion.looks_like_uniform(items)
    sig = "|".join(items[s]["color"] for s in SLOTS)

    c1, c2 = st.columns([1.2, 1])
    occ = c1.selectbox("Occasion", occasion.OCCASIONS, key="occasion")
    uni = c2.checkbox("This is a uniform / fixed dress code", value=detected, key=f"uni_{sig}")
    if detected:
        st.caption(f"Possible uniform detected: {why}. Untick if it isn't one.")
    if uni:
        st.caption("Uniform mode is on, so the occasion above is ignored.")

    a = occasion.analyze(items, occ, uni, res["score"])
    m1, m2, m3 = st.columns(3)
    with m1:
        ui.metric_card("Colour match", f"{res['score']}", "colour theory model")
    with m2:
        ui.metric_card("Uniform check" if uni else "Occasion fit", f"{a['fit']}", f"{a['label']} · {a['occasion']}")
    with m3:
        ui.metric_card("Overall", f"{a['overall']}", "uniform check only" if uni else "colour match + occasion fit")

    st.markdown("##### What we checked")
    for status, text in a["checks"]:
        st.markdown(f"{ICON[status]} {text}")
    for t in a["tips"]:
        ui.html(f'<div class="tip">{t}</div>')
    st.caption(a["note"] + " Advice uses colour, brightness and pattern only. It can't see fit, fabric, logos or your "
               "institution's actual dress code, so treat it as guidance.")