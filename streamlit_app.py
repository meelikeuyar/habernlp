"""
HaberNLP — Editorial Analytics Dashboard
Premium gazete estetigi: NYT / FT / The Economist tarzinda,
tamamen custom HTML/CSS + Plotly ile.

Usage:
    streamlit run streamlit_app.py
"""

import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

# ─────────────────────────────────────────────
# PALETTE — sadece bu dort renk
# ─────────────────────────────────────────────
INK = "#1a1a1a"        # siyah murekkep
PAPER = "#ffffff"      # beyaz
CRIMSON = "#8b0000"    # koyu kirmizi
NEWSPRINT = "#f4f1ec"  # kagit rengi
# turevler (palet icinden opaklik/ton)
INK_60 = "rgba(26,26,26,0.60)"
INK_35 = "rgba(26,26,26,0.35)"
INK_15 = "rgba(26,26,26,0.15)"
INK_08 = "rgba(26,26,26,0.08)"

SENT_COLORS = {"pozitif": INK, "negatif": CRIMSON, "nötr": INK_35}
TR_MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
TR_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

st.set_page_config(
    page_title="HaberNLP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — tek blok
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

/* ── Streamlit kromunu tamamen gizle ── */
#MainMenu, footer, header[data-testid="stHeader"],
.stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {display:none!important;}
.main .block-container{padding-top:1.2rem!important;padding-bottom:0!important;max-width:1240px;}
[data-testid="stAppViewContainer"]{background:#ffffff;}
div[data-testid="stVerticalBlock"]{gap:0.4rem;}

/* ── Genel tipografi ── */
html, body, [class*="css"]{font-family:'Inter',sans-serif;color:#1a1a1a;}

/* ── MASTHEAD ── */
.masthead{text-align:center;padding:26px 0 0;}
.masthead .kicker{font-family:'Inter',sans-serif;font-size:10px;font-weight:500;
  letter-spacing:4px;text-transform:uppercase;color:rgba(26,26,26,0.45);margin-bottom:10px;}
.masthead h1{font-family:'Playfair Display',Georgia,serif;font-size:64px;font-weight:900;
  letter-spacing:-2px;line-height:1;margin:0;color:#1a1a1a;}
.masthead h1 .red{color:#8b0000;}
.masthead .rule-thick{border-top:3px solid #1a1a1a;margin:18px auto 0;width:100%;}
.masthead .rule-thin{border-top:1px solid #1a1a1a;margin:2px auto 0;width:100%;}
.masthead .dateline{display:flex;justify-content:space-between;align-items:center;
  font-family:'Inter',sans-serif;font-size:11px;letter-spacing:2px;text-transform:uppercase;
  color:rgba(26,26,26,0.6);padding:8px 2px;border-bottom:1px solid rgba(26,26,26,0.15);}

/* ── TICKER ── */
.ticker-wrap{background:#1a1a1a;overflow:hidden;white-space:nowrap;margin:14px 0 6px;
  display:flex;align-items:stretch;}
.ticker-label{background:#8b0000;color:#ffffff;font-family:'Inter',sans-serif;font-size:10px;
  font-weight:600;letter-spacing:2px;text-transform:uppercase;padding:9px 16px;flex:0 0 auto;z-index:2;}
.ticker-track{display:inline-block;padding:9px 0;animation:tickermove 60s linear infinite;}
.ticker-track:hover{animation-play-state:paused;}
.ticker-item{font-family:'Playfair Display',Georgia,serif;font-size:13px;color:#ffffff;
  padding:0 28px;display:inline-block;}
.ticker-item .src{font-family:'Inter',sans-serif;font-size:9px;font-weight:600;letter-spacing:1.5px;
  text-transform:uppercase;color:rgba(255,255,255,0.5);margin-right:10px;}
.ticker-sep{color:#8b0000;font-size:13px;}
@keyframes tickermove{0%{transform:translateX(0);}100%{transform:translateX(-50%);}}
@media (prefers-reduced-motion: reduce){.ticker-track{animation:none;}}

/* ── STAT BAND — CSS grid ── */
.stat-band{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid #1a1a1a;margin:20px 0 8px;}
.stat-cell{text-align:center;padding:22px 10px 18px;border-right:1px solid rgba(26,26,26,0.15);}
.stat-cell:last-child{border-right:none;}
.stat-num{font-family:'Playfair Display',Georgia,serif;font-size:42px;font-weight:900;line-height:1;color:#1a1a1a;}
.stat-num.crimson{color:#8b0000;}
.stat-num.faded{color:rgba(26,26,26,0.4);}
.stat-label{font-family:'Inter',sans-serif;font-size:9px;font-weight:600;letter-spacing:2.5px;
  text-transform:uppercase;color:rgba(26,26,26,0.45);margin-top:8px;}
@media (max-width:800px){.stat-band{grid-template-columns:repeat(2,1fr);}
  .stat-cell{border-bottom:1px solid rgba(26,26,26,0.15);}}

/* ── SECTION HEADERS ── */
.sec{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid #1a1a1a;
  padding-bottom:7px;margin:30px 0 14px;}
.sec .t{font-family:'Playfair Display',Georgia,serif;font-size:20px;font-weight:700;color:#1a1a1a;}
.sec .s{font-family:'Inter',sans-serif;font-size:10px;letter-spacing:2px;text-transform:uppercase;
  color:rgba(26,26,26,0.4);}

/* ── NEWS LIST — CSS grid rows ── */
.news-list{display:grid;grid-template-columns:1fr;}
.news-row{display:grid;grid-template-columns:110px 1fr 84px;gap:14px;align-items:baseline;
  padding:11px 2px;border-bottom:1px solid rgba(26,26,26,0.12);}
.news-row:hover{background:#f4f1ec;}
.news-src{font-family:'Inter',sans-serif;font-size:9px;font-weight:600;letter-spacing:1.5px;
  text-transform:uppercase;background:#1a1a1a;color:#ffffff;padding:3px 8px;text-align:center;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.news-headline{font-family:'Playfair Display',Georgia,serif;font-size:15.5px;line-height:1.45;color:#1a1a1a;}
.news-headline a{color:#1a1a1a;text-decoration:none;border-bottom:1px solid transparent;transition:border-color .15s;}
.news-headline a:hover{border-bottom:1px solid #8b0000;color:#8b0000;}
.news-headline a:focus-visible{outline:2px solid #8b0000;outline-offset:2px;}
.news-time{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(26,26,26,0.35);margin-right:8px;}
.news-badge{font-family:'Inter',sans-serif;font-size:8.5px;font-weight:600;letter-spacing:1.5px;
  text-transform:uppercase;text-align:center;padding:3px 0;border:1px solid;}
.b-pozitif{color:#1a1a1a;border-color:#1a1a1a;}
.b-negatif{color:#8b0000;border-color:#8b0000;}
.b-notr{color:rgba(26,26,26,0.4);border-color:rgba(26,26,26,0.25);}
@media (max-width:700px){.news-row{grid-template-columns:1fr;gap:5px;}}

/* ── MODEL SECTION — kagit zemin ── */
.model-band{background:#f4f1ec;border-top:3px solid #1a1a1a;border-bottom:1px solid #1a1a1a;
  padding:28px 30px;margin-top:8px;}
.model-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0;}
.model-cell{text-align:center;padding:6px 10px;border-right:1px solid rgba(26,26,26,0.15);}
.model-cell:last-child{border-right:none;}
.model-num{font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:300;color:#1a1a1a;}
.model-label{font-family:'Inter',sans-serif;font-size:9px;font-weight:600;letter-spacing:2.5px;
  text-transform:uppercase;color:rgba(26,26,26,0.45);margin-top:6px;}
.model-meta{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:rgba(26,26,26,0.55);
  text-align:center;margin-top:18px;letter-spacing:0.5px;}
@media (max-width:700px){.model-grid{grid-template-columns:repeat(2,1fr);}}

/* ── FOOTER ── */
.paper-footer{text-align:center;border-top:1px solid rgba(26,26,26,0.2);margin-top:44px;padding:20px 0 30px;
  font-family:'Inter',sans-serif;font-size:10px;letter-spacing:3px;text-transform:uppercase;
  color:rgba(26,26,26,0.35);}
.paper-footer .dot{color:#8b0000;padding:0 8px;}

/* ── Streamlit widget stillerini editorial hale getir ── */
.stExpander{border:1px solid rgba(26,26,26,0.2)!important;border-radius:0!important;background:#ffffff;}
.stExpander summary{font-family:'Inter',sans-serif!important;font-size:11px!important;
  letter-spacing:2px!important;text-transform:uppercase!important;color:#1a1a1a!important;}
.stExpander details{border-radius:0!important;}
div[data-baseweb="select"]{border-radius:0!important;font-family:'Inter',sans-serif;font-size:13px;}
div[data-baseweb="select"] > div{border-radius:0!important;border-color:rgba(26,26,26,0.3)!important;}
.stTextInput input{border-radius:0!important;border:1px solid rgba(26,26,26,0.3)!important;
  font-family:'Playfair Display',Georgia,serif!important;font-size:15px!important;}
.stTextInput input:focus{border-color:#8b0000!important;box-shadow:none!important;}
.stButton button{border-radius:0!important;border:1px solid #1a1a1a!important;background:#ffffff!important;
  color:#1a1a1a!important;font-family:'Inter',sans-serif!important;font-size:11px!important;
  letter-spacing:2px!important;text-transform:uppercase!important;padding:4px 22px!important;}
.stButton button:hover{background:#1a1a1a!important;color:#ffffff!important;}
.stDateInput input{border-radius:0!important;}
.stMultiSelect [data-baseweb="tag"]{background:#1a1a1a!important;border-radius:0!important;}
.stMultiSelect [data-baseweb="tag"] span{color:#ffffff!important;}
.stCaption, .stCaption p{font-family:'Inter',sans-serif!important;font-size:10px!important;
  letter-spacing:1px;text-transform:uppercase;color:rgba(26,26,26,0.35)!important;}
label[data-testid="stWidgetLabel"] p{font-family:'Inter',sans-serif!important;font-size:10px!important;
  font-weight:600!important;letter-spacing:2px!important;text-transform:uppercase!important;
  color:rgba(26,26,26,0.55)!important;}
[data-testid="stDataFrame"]{font-family:'JetBrains Mono',monospace;}
.stCode, .stCode pre, .stCode code{border-radius:0!important;background:#f4f1ec!important;
  font-family:'JetBrains Mono',monospace!important;font-size:11px!important;color:#1a1a1a!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LAYER (local DB + cloud auto-scrape)
# ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    db_path = Path(__file__).parent / "data" / "habernlp.db"
    if db_path.exists():
        return create_engine(f"sqlite:///{db_path}")
    # Cloud mode: create in-memory DB and auto-populate
    return _cloud_bootstrap()


def _cloud_bootstrap():
    """Scrape fresh news and analyze for cloud deployment."""
    import sqlite3
    db_path = Path(__file__).parent / "data" / "habernlp.db"
    db_path.parent.mkdir(exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    # Create articles table
    with engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text(
                "CREATE TABLE IF NOT EXISTS articles ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "baslik TEXT NOT NULL,"
                "url TEXT,"
                "kaynak TEXT,"
                "tarih DATETIME,"
                "sentiment TEXT,"
                "sentiment_score REAL"
                ")"
            )
        )
        conn.commit()

    # Scrape
    try:
        from src.scraper.scraper import haber_cek
        haberler = haber_cek()
    except Exception:
        haberler = _fallback_scrape()

    # Analyze and insert
    analiz_fn = _get_analiz_fn()
    with engine.connect() as conn:
        for h in haberler:
            label, score = analiz_fn(h["baslik"])
            conn.execute(
                __import__("sqlalchemy").text(
                    "INSERT INTO articles (baslik, url, kaynak, tarih, sentiment, sentiment_score) "
                    "VALUES (:b, :u, :k, :t, :s, :sc)"
                ),
                {"b": h["baslik"], "u": h["url"], "k": h["kaynak"],
                 "t": h["tarih"].isoformat(), "s": label, "sc": score},
            )
        conn.commit()
    return engine


def _get_analiz_fn():
    """Try BERT, fallback to rules."""
    try:
        from src.nlp.sentiment import analiz
        return analiz
    except Exception:
        return _rule_analiz


def _rule_analiz(baslik: str):
    """Lightweight fallback sentiment."""
    _POZ = {"başarı", "zafer", "güzel", "harika", "kazandı", "büyüme", "şampiyon", "rekor", "artış", "umut", "barış"}
    _NEG = {"savaş", "ölüm", "kriz", "patlama", "saldırı", "felaket", "düşüş", "terör", "yangın", "deprem", "cinayet", "gözaltı", "kaza"}
    words = set(re.findall(r"\b\w+\b", baslik.lower()))
    p, n = len(words & _POZ), len(words & _NEG)
    if n > p: return "negatif", -1.0
    if p > n: return "pozitif", 1.0
    return "nötr", 0.0


def _fallback_scrape():
    """Minimal RSS scrape without project imports."""
    import feedparser
    feeds = [
        ("T24", "https://t24.com.tr/rss"),
        ("BBC Türkçe", "https://feeds.bbci.co.uk/turkce/rss.xml"),
        ("NTV", "https://www.ntv.com.tr/gundem.rss"),
        ("Hürriyet", "https://www.hurriyet.com.tr/rss/gundem"),
        ("Sözcü", "https://www.sozcu.com.tr/rss/gundem.xml"),
        ("Habertürk", "https://www.haberturk.com/rss"),
        ("TRT Haber", "https://www.trthaber.com/manset.rss"),
        ("DW Türkçe", "https://rss.dw.com/xml/rss-tur-all"),
    ]
    haberler = []
    for ad, url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:20]:
                haberler.append({"baslik": e.get("title", ""), "url": e.get("link", ""),
                                 "kaynak": ad, "tarih": datetime.now()})
        except Exception:
            pass
    return haberler


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM articles", get_engine())
    if "tarih" in df.columns:
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        df["gun"] = df["tarih"].dt.date
    return df


@st.cache_data(ttl=600)
def load_report():
    p = Path(__file__).parent / "reports" / "training_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def tr_date(dt: datetime) -> str:
    return f"{dt.day} {TR_MONTHS[dt.month - 1]} {dt.year}, {TR_DAYS[dt.weekday()]}"


# ─────────────────────────────────────────────
# PLOTLY HELPERS — palet disina cikma yok
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, sans-serif",
        size=11,
        color=INK
    ),
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    hoverlabel=dict(
        bgcolor=INK,
        font_family="Inter, sans-serif",
        font_color=PAPER,
        font_size=12,
        bordercolor=INK,
    ),
)


def fig_trend(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby(["gun", "sentiment"]).size().unstack(fill_value=0)
    for c in ["pozitif", "negatif", "nötr"]:
        if c not in daily.columns:
            daily[c] = 0
    daily = daily.sort_index()
    fig = go.Figure()
    for name, dash in (("pozitif", "solid"), ("negatif", "solid"), ("nötr", "dot")):
        fig.add_trace(go.Scatter(
            x=list(daily.index), y=daily[name],
            name=name.capitalize(), mode="lines",
            line=dict(color=SENT_COLORS[name], width=1.8, dash=dash),
            hovertemplate=f"<b>{name.capitalize()}</b> %{{y}}<br>%{{x}}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_BASE, height=320, showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=INK_15, ticks="outside",
                     tickcolor=INK_15, tickfont=dict(size=10, color=INK_60))
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=INK_15,
                     tickfont=dict(size=10, color=INK_60))
    return fig


def fig_sources(df: pd.DataFrame) -> go.Figure:
    counts = df["kaynak"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=INK, line=dict(width=0)),
        text=counts.values, textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=INK),
        hovertemplate="<b>%{y}</b> %{x} haber<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_BASE, height=320, bargap=0.45)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(family="Inter, sans-serif", size=11, color=INK))
    return fig


def fig_doughnut(poz: int, neg: int, notr: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Pozitif", "Negatif", "Nötr"],
        values=[poz, neg, notr],
        hole=0.62,
        marker=dict(colors=[INK, CRIMSON, INK_35],
                    line=dict(color=PAPER, width=2)),
        textinfo="percent",
        textfont=dict(family="JetBrains Mono, monospace", size=12, color=PAPER),
        hovertemplate="<b>%{label}</b> %{value} haber (%{percent})<extra></extra>",
        sort=False,
    ))
    total = poz + neg + notr
    fig.update_layout(
        **PLOTLY_BASE, height=320, showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:9px'>HABER</span>",
            showarrow=False,
            font=dict(family="Playfair Display, serif", size=24, color=INK),
        )],
    )
    return fig


def make_wordcloud(texts):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    stop = {
        "ve", "ile", "bu", "bir", "da", "de", "mi", "mu", "mü", "mı", "için",
        "olan", "çok", "daha", "en", "ne", "ki", "şu", "var", "yok", "gibi",
        "kadar", "sonra", "önce", "ama", "ancak", "den", "dan", "nin", "nın",
        "her", "nasıl", "neden", "hangi", "oldu", "olarak", "haber", "haberi",
        "the", "son", "yeni", "etti", "eden", "ise", "diye", "dedi", "değil",
        "ilk", "iki", "üç", "böyle", "şöyle", "artık", "bile", "yine",
    }
    words = re.findall(r"\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{3,}\b", " ".join(texts).lower())
    filtered = [w for w in words if w not in stop]
    if not filtered:
        return None
    wc = WordCloud(
        width=900, height=380, background_color="white", colormap="Greys",
        max_words=60, collocations=False, prefer_horizontal=0.92,
        font_step=1, relative_scaling=0.4,
    ).generate(" ".join(filtered))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    return fig


def sec(title: str, sub: str = ""):
    st.markdown(
        f'<div class="sec"><span class="t">{html.escape(title)}</span>'
        f'<span class="s">{html.escape(sub)}</span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 1. MASTHEAD
# ─────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div class="masthead">
    <div class="kicker">Türkçe Haber Zekâsı · Günlük Bülten</div>
    <h1>Haber<span class="red">NLP</span></h1>
    <div class="rule-thick"></div>
    <div class="rule-thin"></div>
    <div class="dateline">
        <span>{tr_date(now)}</span>
        <span>BERT-Powered Turkish News Intelligence</span>
        <span>Sayı № {now.strftime('%Y%m%d')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

df_all = load_data()
if df_all.empty:
    st.warning("Veritabanında haber yok.")
    st.stop()

# ─────────────────────────────────────────────
# 2. TICKER — en son 15 baslik, sagdan sola
# ─────────────────────────────────────────────
ticker_rows = df_all.sort_values("tarih", ascending=False).head(15)
items = "".join(
    f'<span class="ticker-item"><span class="src">{html.escape(str(r.get("kaynak", "") or ""))}</span>'
    f'{html.escape(str(r.get("baslik", "") or ""))} <span class="ticker-sep">■</span></span>'
    for _, r in ticker_rows.iterrows()
)
st.markdown(f"""
<div class="ticker-wrap">
    <span class="ticker-label">Son Dakika</span>
    <span class="ticker-track">{items}{items}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTERS — expander icinde (icerigi etkiler)
# ─────────────────────────────────────────────
with st.expander("Filtreler ve Arşiv", expanded=False):
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    kaynaklar = sorted(df_all["kaynak"].dropna().unique().tolist())
    with fc1:
        secili_k = st.multiselect("Kaynak", kaynaklar, default=kaynaklar)
    with fc2:
        secili_s = st.multiselect("Duygu", ["pozitif", "negatif", "nötr"],
                                  default=["pozitif", "negatif", "nötr"])
    with fc3:
        d_range = None
        if "gun" in df_all.columns and df_all["gun"].notna().any():
            d_range = st.date_input(
                "Tarih Aralığı",
                value=(df_all["gun"].min(), df_all["gun"].max()),
            )
    with fc4:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("Yenile"):
            st.cache_data.clear()
            st.rerun()

df = df_all[df_all["kaynak"].isin(secili_k)]
df = df[df["sentiment"].isin(secili_s) | df["sentiment"].isna()]
if d_range and isinstance(d_range, tuple) and len(d_range) == 2 and "gun" in df.columns:
    df = df[(df["gun"] >= d_range[0]) & (df["gun"] <= d_range[1])]

if df.empty:
    st.info("Seçilen filtrelerle eşleşen haber yok. Filtreleri genişletin.")
    st.stop()

# ─────────────────────────────────────────────
# 3. STAT BAND
# ─────────────────────────────────────────────
toplam = len(df)
poz = int((df["sentiment"] == "pozitif").sum())
neg = int((df["sentiment"] == "negatif").sum())
notr = int((df["sentiment"] == "nötr").sum())
kaynak_n = df["kaynak"].nunique()

st.markdown(f"""
<div class="stat-band">
    <div class="stat-cell"><div class="stat-num">{toplam:,}</div><div class="stat-label">Toplam Haber</div></div>
    <div class="stat-cell"><div class="stat-num">{poz:,}</div><div class="stat-label">Pozitif</div></div>
    <div class="stat-cell"><div class="stat-num crimson">{neg:,}</div><div class="stat-label">Negatif</div></div>
    <div class="stat-cell"><div class="stat-num faded">{notr:,}</div><div class="stat-label">Nötr</div></div>
    <div class="stat-cell"><div class="stat-num">{kaynak_n}</div><div class="stat-label">Kaynak</div></div>
</div>
""".replace(",", "."), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. ANA ICERIK — trend + kaynak dagilimi
# ─────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    sec("Günlük Duygu Trendi", "Sentiment / Gün")
    if "gun" in df.columns and df["gun"].notna().any():
        st.plotly_chart(fig_trend(df), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.caption("Tarih verisi bulunamadı")
with c2:
    sec("Kaynak Dağılımı", "Haber / Kaynak")
    st.plotly_chart(fig_sources(df), use_container_width=True,
                    config={"displayModeBar": False})

# ─────────────────────────────────────────────
# 5. IKINCI SATIR — kelime bulutu + doughnut
# ─────────────────────────────────────────────
c3, c4 = st.columns([3, 2], gap="large")
with c3:
    sec("Kelime Bulutu", "Başlıklardan")
    basliklar = df["baslik"].dropna().astype(str).tolist()
    fig_wc = make_wordcloud(basliklar) if basliklar else None
    if fig_wc:
        st.pyplot(fig_wc)
    else:
        st.caption("Kelime bulutu için: pip install wordcloud")
with c4:
    sec("Duygu Oranları", "Dağılım")
    st.plotly_chart(fig_doughnut(poz, neg, notr), use_container_width=True,
                    config={"displayModeBar": False})

# ─────────────────────────────────────────────
# 6. HABER LISTESI — manset listesi
# ─────────────────────────────────────────────
sec("Manşetler", f"Son {min(30, len(df))} haber")
arama = st.text_input("Arama", placeholder="Başlıkta ara...",
                      label_visibility="collapsed")
df_list = df
if arama:
    df_list = df_list[df_list["baslik"].str.contains(arama, case=False, na=False)]

news_df = df_list.sort_values("tarih", ascending=False).head(30)
rows_html = []
badge_map = {"pozitif": "b-pozitif", "negatif": "b-negatif", "nötr": "b-notr"}
for _, row in news_df.iterrows():
    s = str(row.get("sentiment") or "")
    bc = badge_map.get(s, "b-notr")
    bt = html.escape(s.upper()) if s else "—"
    ts = row["tarih"].strftime("%H:%M") if pd.notna(row.get("tarih")) else "--:--"
    kaynak = html.escape(str(row.get("kaynak") or ""))
    baslik = html.escape(str(row.get("baslik") or ""))
    url = str(row.get("url") or "").strip()
    if url.startswith("http"):
        headline = (f'<a href="{html.escape(url)}" target="_blank" '
                    f'rel="noopener noreferrer">{baslik}</a>')
    else:
        headline = baslik
    rows_html.append(
        f'<div class="news-row">'
        f'<span class="news-src">{kaynak}</span>'
        f'<span class="news-headline"><span class="news-time">{ts}</span>{headline}</span>'
        f'<span class="news-badge {bc}">{bt}</span>'
        f'</div>'
    )
st.markdown(f'<div class="news-list">{"".join(rows_html)}</div>',
            unsafe_allow_html=True)
st.caption(f"{len(news_df)} / {len(df_list)} haber gösteriliyor")

# ─────────────────────────────────────────────
# 7. MODEL PERFORMANSI — kagit zemin
# ─────────────────────────────────────────────
sec("Model Performansı", "Test Kümesi Metrikleri")
report = load_report()
if report:
    tm = report.get("test_metrics", {})

    def fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "—"

    st.markdown(f"""
    <div class="model-band">
        <div class="model-grid">
            <div class="model-cell"><div class="model-num">{fmt(tm.get('f1_macro'))}</div><div class="model-label">F1 Macro</div></div>
            <div class="model-cell"><div class="model-num">{fmt(tm.get('accuracy'))}</div><div class="model-label">Accuracy</div></div>
            <div class="model-cell"><div class="model-num">{fmt(tm.get('precision_macro'))}</div><div class="model-label">Precision</div></div>
            <div class="model-cell"><div class="model-num">{fmt(tm.get('recall_macro'))}</div><div class="model-label">Recall</div></div>
        </div>
        <div class="model-meta">
            {html.escape(str(report.get('base_model', '—')))} ·
            {html.escape(str(report.get('dataset_size', '—')))} örnek ·
            train/val/test {html.escape(str(report.get('splits', {}).get('train', '—')))}/{html.escape(str(report.get('splits', {}).get('val', '—')))}/{html.escape(str(report.get('splits', {}).get('test', '—')))} ·
            {html.escape(str(report.get('timestamp', ''))[:16])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2, gap="large")
    with ec1:
        with st.expander("Confusion Matrix"):
            cm = report.get("confusion_matrix")
            if cm:
                labels = ["negatif", "nötr", "pozitif"]
                st.dataframe(
                    pd.DataFrame(cm, index=labels, columns=labels),
                    use_container_width=True,
                )
            else:
                st.caption("Confusion matrix bulunamadı")
        with st.expander("Hiperparametreler"):
            hp = report.get("hyperparameters", {})
            if hp:
                st.markdown("\n\n".join(f"**{html.escape(str(k))}:** `{html.escape(str(v))}`"
                                        for k, v in hp.items()))
            else:
                st.caption("Hiperparametre bilgisi yok")
    with ec2:
        with st.expander("Classification Report"):
            cr = report.get("classification_report", "")
            if cr:
                st.code(cr, language=None)
            else:
                st.caption("Classification report bulunamadı")
    run_id = str(report.get("mlflow_run_id") or "")
    if run_id:
        st.caption(f"MLflow Run: {run_id[:12]}")
else:
    st.markdown("""
    <div class="model-band">
        <div class="model-meta">Eğitim raporu bulunamadı — reports/training_report.json oluşturmak için modeli eğitin.</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="paper-footer">
    HaberNLP<span class="dot">·</span>BERT-Powered Turkish News Intelligence<span class="dot">·</span>Est. 2026
</div>
""", unsafe_allow_html=True)