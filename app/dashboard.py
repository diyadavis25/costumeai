"""Member 3 - Streamlit pages (Home, Analyze, AI Analysis, Recommendations, Wardrobe, Performance, About)."""
import hashlib
import json

import pandas as pd
import streamlit as st

from app import charts, ui
from src import compatibility
from src.clothing_analysis import analyze_clothing
from src.clothing_classifier import get_classifier
from src.compatibility import predict_compatibility
from src.config import METRICS_PATH, PALETTE, SLOTS, SLOT_TITLE
from src.feature_extraction import get_extractor
from src.feedback import feedback_count, save_feedback
from src.preprocessing import image_to_bytes, load_image
from src.recommendation import rank_wardrobe, recommend_alternatives
from src.sample_garments import garment_bytes

PAGES = ["Home", "Analyze Outfit", "AI Analysis", "Recommendations", "My Wardrobe",
         "Model Performance", "About"]
DEMO = {"top": "white", "bottom": "blue", "shoes": "red"}


# ---------------------------------------------------------------- state / helpers
def init_state():
    ss = st.session_state
    ss.setdefault("nav", "Home")
    ss.setdefault("outfit", {s: None for s in SLOTS})
    ss.setdefault("wardrobe", {s: [] for s in SLOTS})
    ss.setdefault("w_seen", set())
    ss.setdefault("w_results", None)
    ss.setdefault("wkey", 0)


def go(page):
    st.session_state.nav = page


@st.cache_resource(show_spinner=False)
def get_compat_model():
    return compatibility.get_model()


@st.cache_data(show_spinner=False)
def _analyze(data: bytes, slot: str):
    return analyze_clothing(data, slot)


def make_item(data: bytes, slot: str) -> dict:
    item = dict(_analyze(data, slot))
    item["thumb"] = image_to_bytes(load_image(data))
    item["uid"] = hashlib.sha1(data).hexdigest()[:10] + slot
    item["manual"] = False
    return item


def recolor(item, color):
    item.update(color=color, rgb=PALETTE[color], hex=ui.hexrgb(PALETTE[color]), manual=True)


def load_demo():
    for s, c in DEMO.items():
        st.session_state.outfit[s] = make_item(garment_bytes(s, c), s)
    go("AI Analysis")


def load_sample_wardrobe():
    sets = {"top": ["white", "black", "navy", "beige", "red"],
            "bottom": ["blue", "black", "beige", "grey", "olive"],
            "shoes": ["white", "black", "brown", "red", "grey"]}
    for s, colors in sets.items():
        for i, c in enumerate(colors):
            data = garment_bytes(s, c, pattern=(s == "top" and c == "red"))
            h = hashlib.sha1(data).hexdigest()[:10] + s
            if h not in st.session_state.w_seen:
                st.session_state.w_seen.add(h)
                st.session_state.wardrobe[s].append(make_item(data, s))


def clear_wardrobe():
    st.session_state.wardrobe = {s: [] for s in SLOTS}
    st.session_state.w_seen = set()
    st.session_state.w_results = None
    st.session_state.wkey += 1


def outfit_ready():
    return all(st.session_state.outfit[s] for s in SLOTS)


def empty_state(msg="Analyze an outfit first."):
    st.info(msg)
    st.button("Go to Analyze Outfit", on_click=go, args=("Analyze Outfit",))


def current_result():
    o = st.session_state.outfit
    return predict_compatibility(o["top"], o["bottom"], o["shoes"], get_compat_model())


# ---------------------------------------------------------------- Home
def page_home():
    ui.hero()
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("Start analyzing", type="primary", on_click=go, args=("Analyze Outfit",))
    c2.button("Try the demo outfit", on_click=load_demo)
    st.write("")
    cols = st.columns(3)
    cards = [("See what you are wearing", "ResNet18 features, K-Means dominant colour and category detection from a photo."),
             ("Get a compatibility score", "A model trained on colour, contrast and busyness scores top + bottom + shoes from 0 to 100."),
             ("Know what to change", "The engine tries every alternative colour, and ranks your whole wardrobe's best outfits.")]
    for col, (t, d) in zip(cols, cards):
        with col:
            ui.html(f'<div class="card"><h4>{t}</h4><p class="muted">{d}</p></div>')
    section_status()


def section_status():
    ui.section("System status")
    clf = get_classifier()
    metrics = _load_metrics()
    cols = st.columns(4)
    with cols[0]:
        ui.metric_card("Embedding backend", get_extractor().backend, "512-d visual features")
    with cols[1]:
        ui.metric_card("Category classifier", "ResNet18" if clf.available else "slot hint",
                       "trained weights loaded" if clf.available else "train it with training/train_classifier.py")
    with cols[2]:
        ui.metric_card("Compatibility model", get_compat_model().name.split(" (")[0], "selected on validation F1")
    with cols[3]:
        ui.metric_card("Feedback collected", feedback_count(), "👍/👎 labels for retraining")


# ---------------------------------------------------------------- Analyze
def slot_input(slot):
    st.markdown(f"#### {SLOT_TITLE[slot]}")
    src = st.radio("Source", ["Upload", "Camera", "Sample"], horizontal=True, key=f"src_{slot}",
                   label_visibility="collapsed")
    data = None
    if src == "Upload":
        f = st.file_uploader(f"Upload {slot}", type=["jpg", "jpeg", "png", "webp"], key=f"up_{slot}",
                             label_visibility="collapsed")
        data = f.getvalue() if f else None
    elif src == "Camera":
        c = st.camera_input(f"Take a {slot} photo", key=f"cam_{slot}", label_visibility="collapsed")
        data = c.getvalue() if c else None
        st.caption("Hold the garment close to the lens against a plain background for best colour reading.")
    else:
        names = list(PALETTE)
        color = st.selectbox("Sample colour", names, index=names.index(DEMO[slot]), key=f"samp_{slot}")
        pattern = st.checkbox("Striped", key=f"pat_{slot}")
        data = garment_bytes(slot, color, pattern)
    if data:
        st.image(data, width=210)
    return data


def page_analyze():
    ui.section("Analyze your outfit", "Add each piece by upload, camera, or a built-in sample, then run the analysis.")
    cols = st.columns(3)
    data = {}
    for col, slot in zip(cols, SLOTS):
        with col:
            data[slot] = slot_input(slot)
    ready = all(data.values())
    if st.button("Analyze outfit", type="primary", disabled=not ready):
        with st.spinner("Reading colours, extracting features, scoring..."):
            for s in SLOTS:
                st.session_state.outfit[s] = make_item(data[s], s)
    if not ready:
        st.caption("Add all three pieces (top, bottom, shoes) to enable analysis.")
    if outfit_ready():
        st.divider()
        res = current_result()
        c1, c2 = st.columns([1, 2])
        with c1:
            ui.score_ring(res["score"], res["label"])
        with c2:
            ui.html("".join(ui.color_chip(f"{st.session_state.outfit[s]['color']} {s}",
                                          st.session_state.outfit[s]["rgb"]) for s in SLOTS))
            for t in res["tips"]:
                ui.html(f'<div class="tip">{t}</div>')
            st.button("See the full AI analysis", on_click=go, args=("AI Analysis",))


# ---------------------------------------------------------------- AI analysis
def item_card(slot, item):
    st.markdown(f"#### {SLOT_TITLE[slot]}")
    st.image(item["thumb"], width=200)
    if item["category_source"] == "classifier":
        cat = f"{item['category']} · {item['confidence'] * 100:.0f}% confidence"
    else:
        cat = f"{item['category']} (from slot; train the classifier to auto-detect)"
    pal = "".join(ui.color_chip(p["name"], p["rgb"]) for p in item["palette"])
    ui.html(f'<div class="card"><h4>{item["color"].title()} {slot}</h4><div class="muted">{cat}</div>'
            f'<div style="margin-top:8px">{pal}</div>'
            f'<div class="muted" style="margin-top:6px">Busyness {item["busyness"]:.2f}'
            f'{" · colour corrected by you" if item["manual"] else ""}</div></div>')
    names = list(PALETTE)
    new = st.selectbox("Colour wrong? Fix it", names, index=names.index(item["color"]) if item["color"] in names else 0,
                       key=f"fix_{item['uid']}")
    if new != item["color"]:
        recolor(item, new)


def page_analysis():
    if not outfit_ready():
        return empty_state()
    ui.section("AI analysis", "What the vision pipeline detected, and how the model rates the combination.")
    cols = st.columns(3)
    for col, slot in zip(cols, SLOTS):
        with col:
            item_card(slot, st.session_state.outfit[slot])
    res = current_result()
    st.divider()
    c1, c2 = st.columns([1, 1.5])
    with c1:
        ui.score_ring(res["score"], res["label"], size=190)
        st.caption(f"Model: {res['model']}")
    with c2:
        st.markdown("#### Why this score")
        for k, v in res["breakdown"].items():
            ui.bar_row(k, v)
        st.caption("Colour Harmony and Visual Balance come from transparent styling rules; Visual Similarity is the "
                   "cosine similarity of the 512-d embeddings (informational). Category Match needs the trained classifier.")
    for t in res["tips"]:
        ui.html(f'<div class="tip">{t}</div>')
    st.markdown("##### Was this useful? Your vote becomes training data.")
    f1, f2, f3 = st.columns([1, 1, 3])
    if f1.button("👍 Looks great"):
        n = save_feedback(st.session_state.outfit, True)
        f3.success(f"Saved. {n} votes collected.")
    if f2.button("👎 Not for me"):
        n = save_feedback(st.session_state.outfit, False)
        f3.success(f"Saved. {n} votes collected.")
    st.button("Get recommendations", type="primary", on_click=go, args=("Recommendations",))


# ---------------------------------------------------------------- Recommendations
def apply_change(slot, color):
    recolor(st.session_state.outfit[slot], color)


def page_recommendations():
    if not outfit_ready():
        return empty_state()
    o = st.session_state.outfit
    ui.section("Recommended alternatives", "Every palette colour is tried in each slot and scored by the model.")
    rec = recommend_alternatives(o["top"], o["bottom"], o["shoes"], get_compat_model())
    c1, c2 = st.columns([1, 2])
    with c1:
        ui.score_ring(rec["current_score"], "Current outfit", size=140)
    with c2:
        best = rec["slots"][rec["best_slot"]][0]
        gain = best["delta"]
        if gain > 0:
            ui.html(f'<div class="card"><h4>Best single change: {best["item_name"]}</h4>'
                    f'<p class="muted">Swap your {rec["best_slot"]} and the score moves from {rec["current_score"]} '
                    f'to <b>{best["score"]}</b> (+{gain}).</p></div>')
        else:
            ui.html('<div class="card"><h4>Nothing beats this outfit</h4><p class="muted">No single colour swap improves the score. Nice.</p></div>')
    tabs = st.tabs([f"Change {SLOT_TITLE[s].lower()}" for s in SLOTS])
    for tab, slot in zip(tabs, SLOTS):
        with tab:
            for i, alt in enumerate(rec["slots"][slot]):
                col = ui.MINT if alt["delta"] > 0 else ui.CORAL
                a, b = st.columns([4, 1])
                with a:
                    ui.html(f'<div class="rank"><div class="medal">{ui.medal(i)}</div>'
                            f'<span class="sw" style="width:26px;height:26px;background:{ui.hexrgb(alt["rgb"])}"></span>'
                            f'<div class="nm">{alt["item_name"]}</div>'
                            f'<div class="sc">{alt["score"]}%<span class="delta" style="color:{col}">{alt["delta"]:+d}</span></div></div>')
                with b:
                    st.button("Apply", key=f"apply_{slot}_{alt['color']}", on_click=apply_change, args=(slot, alt["color"]))
    st.button("Back to analysis", on_click=go, args=("AI Analysis",))


# ---------------------------------------------------------------- Wardrobe
def add_to_wardrobe(slot, data):
    h = hashlib.sha1(data).hexdigest()[:10] + slot
    if h in st.session_state.w_seen:
        return
    st.session_state.w_seen.add(h)
    st.session_state.wardrobe[slot].append(make_item(data, slot))


def page_wardrobe():
    ui.section("My wardrobe", "Add several tops, bottoms and shoes. Every combination is scored and the best outfits are ranked.")
    b1, b2, _ = st.columns([1.3, 1, 2])
    b1.button("Load sample wardrobe (5 / 5 / 5)", on_click=load_sample_wardrobe)
    b2.button("Clear wardrobe", on_click=clear_wardrobe)
    tabs = st.tabs(["Tops", "Bottoms", "Shoes"])
    k = st.session_state.wkey
    for tab, slot in zip(tabs, SLOTS):
        with tab:
            files = st.file_uploader(f"Upload {slot} photos", type=["jpg", "jpeg", "png", "webp"],
                                     accept_multiple_files=True, key=f"wup_{slot}_{k}")
            for f in files or []:
                add_to_wardrobe(slot, f.getvalue())
            with st.expander("Add with camera"):
                cam = st.camera_input(f"Take a {slot} photo", key=f"wcam_{slot}_{k}")
                if cam:
                    add_to_wardrobe(slot, cam.getvalue())
            items = st.session_state.wardrobe[slot]
            st.caption(f"{len(items)} item(s)")
            for row in range(0, len(items), 6):
                cols = st.columns(6)
                for col, it in zip(cols, items[row:row + 6]):
                    with col:
                        st.image(it["thumb"], width=100, caption=it["color"])
    counts = [len(st.session_state.wardrobe[s]) for s in SLOTS]
    total = counts[0] * counts[1] * counts[2]
    st.divider()
    st.markdown(f"#### Generate outfits  ·  {counts[0]} × {counts[1]} × {counts[2]} = **{total}** combinations")
    c1, c2, c3 = st.columns([1, 1, 1])
    top_n = c1.slider("How many to show", 3, 10, 5)
    diverse = c2.checkbox("Avoid near-duplicates", value=True)
    if c3.button("Generate outfits", type="primary", disabled=total == 0):
        best, scores = rank_wardrobe(st.session_state.wardrobe, get_compat_model(), top_n, diverse)
        st.session_state.w_results = (best, scores)
    if st.session_state.w_results and total:
        best, scores = st.session_state.w_results
        ui.section("Your best outfits")
        for i, o in enumerate(best):
            ti, bi, si = o["idx"]
            try:
                parts = [st.session_state.wardrobe[s][j] for s, j in zip(SLOTS, (ti, bi, si))]
            except IndexError:
                continue
            cols = st.columns([1, 1, 1, 1.3])
            for col, p, s in zip(cols[:3], parts, SLOTS):
                with col:
                    st.image(p["thumb"], width=120, caption=f"{p['color']} {s}")
            with cols[3]:
                ui.html(f'<div style="font-size:1.8rem">{ui.medal(i)}</div>')
                ui.score_ring(o["score"], o["label"], size=100)
            st.divider()
        charts.show(charts.score_hist(scores))


# ---------------------------------------------------------------- Performance
def _load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def retrain():
    from src.model_training import train_and_save
    with st.spinner("Training models..."):
        train_and_save(verbose=False)
    compatibility._MODEL = None
    st.cache_resource.clear()


def page_performance():
    ui.section("Model performance", "Held-out test results for the three compatibility models, measured by the training script.")
    m = _load_metrics()
    if not m:
        st.warning("No metrics found yet.")
        st.button("Train models now", on_click=retrain)
        return
    sel = m["selected_model"]
    t = m["models"][sel]["test"]
    st.markdown(f"**Selected model:** {sel}  ·  rule: {m['selection_rule']}")
    cols = st.columns(5)
    for col, (lab, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                      ("F1 score", "f1"), ("ROC-AUC", "roc_auc")]):
        with col:
            ui.metric_card(lab, f"{t[key] * 100:.1f}%" if key != "roc_auc" else f"{t[key]:.3f}", "test set")
    st.write("")
    rows = []
    for n, v in m["models"].items():
        rows.append({"Model": n, "Val F1": round(v["validation"]["f1"], 4), "Test Accuracy": round(v["test"]["accuracy"], 4),
                     "Test Precision": round(v["test"]["precision"], 4), "Test Recall": round(v["test"]["recall"], 4),
                     "Test F1": round(v["test"]["f1"], 4), "Test ROC-AUC": round(v["test"]["roc_auc"], 4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        charts.show(charts.confusion(m))
    with c2:
        charts.show(charts.roc(m))
    charts.show(charts.comparison(m))
    charts.show(charts.importance(m))
    d = m["dataset"]
    st.caption(f"Data: {d['rows_total']} outfits (train {d['train']} / val {d['validation']} / test {d['test']}), "
               f"{d['positive_rate'] * 100:.0f}% compatible, {d['user_feedback_rows']} feedback rows. "
               f"Majority-class baseline accuracy: {m['baseline_majority_accuracy'] * 100:.1f}%.")
    ui.html('<div class="note"><b>Read these numbers honestly.</b> The training labels come from transparent colour-harmony rules '
            'plus noise, because no free colour-based outfit label set exists offline. The scores show how well the models '
            'learn that rule set, not how well they predict human taste. Collect 👍/👎 votes in the app, or add real '
            'labelled data to <code>data/outfit_pairs.csv</code>, and retrain.</div>')
    if not m.get("xgboost_available", True):
        st.caption("XGBoost is not installed here, so a scikit-learn gradient-boosting model stands in. Run "
                   "`pip install xgboost` and retrain to include the real XGBoost.")
    st.button("Retrain models", on_click=retrain)


# ---------------------------------------------------------------- About
def page_about():
    ui.section("About Outfit AI", "Three-member capstone: vision, machine learning, and product.")
    cols = st.columns(3)
    team = [("Member 1 · Computer vision", "Dataset prep, ResNet18 category classifier, K-Means colour, 512-d features.",
             "src/clothing_analysis.py · color_features.py · feature_extraction.py"),
            ("Member 2 · Compatibility ML", "Outfit dataset, feature engineering, Logistic Regression / Random Forest / XGBoost comparison.",
             "src/compatibility.py · model_training.py · training/"),
            ("Member 3 · Recommendation + dashboard", "Alternative suggestions, wardrobe ranking, this Streamlit app.",
             "src/recommendation.py · app/")]
    for col, (t, d, f) in zip(cols, team):
        with col:
            ui.html(f'<div class="card"><h4>{t}</h4><p class="muted">{d}</p><code style="font-size:.75rem">{f}</code></div>')
    st.markdown("""
**Pipeline:** image → preprocessing → ResNet18 features + K-Means colour (+ category) → outfit feature vector → compatibility model → score, breakdown, alternatives, wardrobe ranking.

**Limits worth stating at the expo**
- Compatibility labels are rule-generated (with noise), not human ratings. Score bands (80 / 65 / 50) are application-defined.
- Colour is read from the dominant foreground cluster; busy backgrounds or strong lighting can fool it. Use the colour-fix box.
- Without PyTorch and trained weights the app uses a handcrafted 512-d descriptor and takes the category from the slot you uploaded into.
- This is a styling aid, not an objective measure of how an outfit suits a person.
""")
