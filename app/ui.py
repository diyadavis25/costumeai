"""Member 3 - reusable UI components + theme (HTML/CSS injected into Streamlit)."""
import streamlit as st

MINT, VIOLET, AMBER, CORAL = "#19B37F", "#5A3CF0", "#F5A524", "#FF6B57"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--ink:#15151F;--paper:#F4F2FA;--card:#fff;--violet:#5A3CF0;--soft:#E4DEFF;--line:#E3E0F0;--muted:#6B6880}
.stApp{background:var(--paper)}
html,body,.stApp,p,li,label,span,div{font-family:'DM Sans',system-ui,sans-serif}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','DM Sans',sans-serif!important;letter-spacing:-.02em;color:var(--ink)}
section[data-testid="stSidebar"]{background:var(--ink)}
section[data-testid="stSidebar"] *{color:#EDEBFA!important}
.block-container{padding-top:2.2rem;max-width:1180px}
.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:36px;align-items:center;margin:6px 0 26px}
.hero h1{font-size:3.4rem;line-height:1.02;margin:0 0 14px;font-weight:800}
.hero p{font-size:1.1rem;color:var(--muted);max-width:34ch;line-height:1.55}
.mirror{background:linear-gradient(160deg,#fff 0%,#DCD5FF 55%,#B9AAFF 100%);border:10px solid var(--ink);border-radius:999px 999px 28px 28px;padding:42px 26px 26px;text-align:center;box-shadow:0 24px 50px rgba(40,25,120,.25)}
.mirror .swatches{display:flex;justify-content:center;gap:10px;margin:14px 0 6px}
.sw{width:46px;height:46px;border-radius:50%;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.18)}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 20px;margin-bottom:14px}
.card h4{margin:0 0 4px;font-size:1.05rem}
.muted{color:var(--muted);font-size:.92rem}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--soft);color:var(--ink);padding:4px 11px 4px 5px;border-radius:99px;font-size:.85rem;font-weight:500;margin:2px 6px 2px 0}
.chip i{display:inline-block;width:16px;height:16px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 0 1px #0002}
.tag{display:inline-block;padding:3px 11px;border-radius:99px;font-size:.8rem;font-weight:700;color:#fff}
.ring{width:var(--sz,160px);height:var(--sz,160px);border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),#E6E3F2 0)}
.ring-in{width:calc(var(--sz,160px) - 30px);height:calc(var(--sz,160px) - 30px);border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring-in b{font-family:'Bricolage Grotesque',sans-serif;font-size:calc(var(--sz,160px)*.32);line-height:1}
.ring-in span{font-size:.75rem;color:var(--muted)}
.bar{margin:10px 0}.bar .row{display:flex;justify-content:space-between;font-size:.92rem;margin-bottom:4px}
.bar .track{height:10px;background:#ECE9F6;border-radius:99px;overflow:hidden}
.bar .fill{height:100%;border-radius:99px}
.rank{display:flex;align-items:center;gap:12px;padding:10px 14px;border:1px solid var(--line);border-radius:14px;background:#fff;margin-bottom:8px}
.rank .medal{font-size:1.5rem;width:34px;text-align:center}
.rank .nm{flex:1;font-weight:700}
.rank .sc{font-family:'Bricolage Grotesque',sans-serif;font-size:1.35rem;font-weight:800}
.delta{font-size:.8rem;font-weight:700;margin-left:6px}
.metric{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 16px}
.metric b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.9rem}
.metric span{color:var(--muted);font-size:.85rem}
.note{background:#FFF6E0;border:1px solid #F3DDA0;border-radius:14px;padding:12px 16px;font-size:.9rem}
.tip{border-left:4px solid var(--violet);background:#fff;padding:10px 14px;border-radius:0 12px 12px 0;margin:8px 0}
.brand{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.5rem;letter-spacing:-.03em}
.stButton>button{border-radius:12px;font-weight:700}
@media(max-width:800px){.hero{grid-template-columns:1fr}.hero h1{font-size:2.4rem}}
</style>
"""


def html(s: str):
    st.markdown(" ".join(s.split()), unsafe_allow_html=True)


def inject_css():
    html(CSS)


def score_color(score):
    return MINT if score >= 80 else VIOLET if score >= 65 else AMBER if score >= 50 else CORAL


def hexrgb(rgb):
    return "#%02x%02x%02x" % tuple(int(x) for x in rgb)


def score_ring(score, label=None, size=160):
    c = score_color(score)
    out = f'<div class="ring" style="--p:{score};--c:{c};--sz:{size}px"><div class="ring-in"><b>{score}</b><span>/ 100</span></div></div>'
    if label:
        out += f'<div style="margin-top:10px"><span class="tag" style="background:{c}">{label}</span></div>'
    html(f'<div style="display:flex;flex-direction:column;align-items:center">{out}</div>')


def bar_row(name, value):
    if value is None:
        html(f'<div class="bar"><div class="row"><span>{name}</span><span class="muted">n/a</span></div></div>')
        return
    html(f'<div class="bar"><div class="row"><span>{name}</span><b>{value}%</b></div>'
         f'<div class="track"><div class="fill" style="width:{value}%;background:{score_color(value)}"></div></div></div>')


def color_chip(name, rgb):
    return f'<span class="chip"><i style="background:{hexrgb(rgb)}"></i>{name.title()}</span>'


def metric_card(label, value, hint=""):
    html(f'<div class="metric"><span>{label}</span><b>{value}</b><span>{hint}</span></div>')


def section(title, sub=""):
    st.markdown(f"## {title}")
    if sub:
        html(f'<p class="muted" style="margin-top:-8px">{sub}</p>')


def hero():
    html("""
    <div class="hero"><div>
      <h1>Does this outfit actually work?</h1>
      <p>Snap or upload your top, bottom and shoes. Outfit AI reads colour and shape, scores the combination,
      and tells you exactly what to swap.</p></div>
      <div class="mirror"><div class="muted">Mirror says</div>
        <div style="display:flex;justify-content:center;margin-top:8px"><div class="ring" style="--p:87;--c:#19B37F;--sz:150px"><div class="ring-in"><b>87</b><span>/ 100</span></div></div></div>
        <div class="swatches"><div class="sw" style="background:#F5F5F5"></div><div class="sw" style="background:#2850C8"></div><div class="sw" style="background:#F5F5F5"></div></div>
        <span class="tag" style="background:#19B37F">Excellent Match</span></div></div>""")


def medal(i):
    return ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."
