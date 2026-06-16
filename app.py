"""
Goodreads Recommender — Fixed UX
Widgets sit inside styled columns that form the book pages
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field

from surprise import KNNBasic, Dataset, Reader
from google import genai

st.set_page_config(
    page_title="Goodreads Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "phase" not in st.session_state:
    st.session_state.phase = "intro"
if "cf_recs" not in st.session_state:
    st.session_state.cf_recs = []
if "reranked" not in st.session_state:
    st.session_state.reranked = []

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
section[data-testid="stSidebar"] { display: none !important; }

/* Remove all default Streamlit padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] { gap: 0rem !important; }

/* ── Animated library background ── */
.library-bg {
    position: fixed;
    inset: 0;
    background: linear-gradient(160deg, #1a0a00 0%, #3d1a00 30%, #5c2d00 55%, #2a0d00 100%);
    overflow: hidden;
    z-index: 0;
}
.shelf {
    position: absolute;
    left: 0; right: 0;
    height: 110px;
    display: flex;
    align-items: flex-end;
    gap: 3px;
    padding: 0 20px;
    border-bottom: 8px solid #3a1f00;
    animation: shelfFloat 8s ease-in-out infinite;
}
.shelf:nth-child(1) { bottom: 0px;   animation-delay: 0s; }
.shelf:nth-child(2) { bottom: 115px; animation-delay: 1s; }
.shelf:nth-child(3) { bottom: 230px; animation-delay: 2s; }
.shelf:nth-child(4) { bottom: 345px; animation-delay: 0.5s; }
.shelf:nth-child(5) { bottom: 460px; animation-delay: 1.5s; }
@keyframes shelfFloat {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-4px); }
}
.book-spine {
    width: 28px;
    border-radius: 3px 3px 0 0;
    box-shadow: inset -3px 0 6px rgba(0,0,0,0.4);
}
.particle {
    position: absolute;
    border-radius: 50%;
    background: rgba(255,210,100,0.5);
    animation: drift linear infinite;
}
@keyframes drift {
    0%   { transform: translateY(100vh); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.4; }
    100% { transform: translateY(-20px) translateX(20px); opacity: 0; }
}
.glow-overlay {
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at 50% 60%, rgba(255,160,50,0.07) 0%, transparent 70%);
    pointer-events: none; z-index: 1;
    animation: flicker 4s ease-in-out infinite;
}
@keyframes flicker { 0%,100%{opacity:1;} 50%{opacity:0.75;} }

/* ── Intro screen ── */
.intro-screen {
    position: relative; z-index: 10;
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 32px;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700; color: #f5d78e;
    text-align: center;
    text-shadow: 0 0 40px rgba(255,180,50,0.5), 2px 4px 8px rgba(0,0,0,0.8);
    line-height: 1.15;
    animation: fadeUp 1.8s ease forwards;
}
.main-title em { font-style: italic; color: #ffeaa0; }
.subtitle {
    font-family: 'Lato', sans-serif; font-size: 1rem;
    font-weight: 300; color: rgba(245,215,142,0.65);
    letter-spacing: 0.22em; text-transform: uppercase;
    animation: fadeUp 2.2s ease forwards;
}
@keyframes fadeUp {
    0%   { opacity: 0; transform: translateY(-24px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ── Book pages: style the Streamlit COLUMNS directly ── */
/* Left page column */
[data-testid="stHorizontalBlock"] > div:nth-child(1) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(160deg, #fdf6e3 0%, #f9edd8 50%, #f5e6cc 100%) !important;
    min-height: 92vh !important;
    padding: 36px 32px 28px !important;
    border-radius: 6px 0 0 6px !important;
    box-shadow: -4px 0 24px rgba(0,0,0,0.5), inset -6px 0 16px rgba(0,0,0,0.06) !important;
    position: relative !important;
    z-index: 10 !important;
    background-image:
        linear-gradient(160deg, #fdf6e3 0%, #f9edd8 50%, #f5e6cc 100%),
        repeating-linear-gradient(transparent, transparent 27px, rgba(180,120,40,0.07) 27px, rgba(180,120,40,0.07) 28px) !important;
}

/* Spine column */
[data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(180deg, #5a2a00, #3d1500, #5a2a00) !important;
    min-height: 92vh !important;
    box-shadow: -5px 0 14px rgba(0,0,0,0.5), 5px 0 14px rgba(0,0,0,0.5) !important;
    z-index: 11 !important;
}

/* Right page column */
[data-testid="stHorizontalBlock"] > div:nth-child(3) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(200deg, #fdf6e3 0%, #f9edd8 50%, #f5e6cc 100%) !important;
    min-height: 92vh !important;
    padding: 36px 32px 28px !important;
    border-radius: 0 6px 6px 0 !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5), inset 6px 0 16px rgba(0,0,0,0.06) !important;
    position: relative !important;
    z-index: 10 !important;
}

/* ── Book page wrapper (z-index above library) ── */
.book-wrapper {
    position: relative;
    z-index: 10;
    padding: 16px 12px;
}

/* ── Page typography ── */
.page-chapter {
    font-family: 'Lato', sans-serif;
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #8B4513; margin-bottom: 6px;
}
.page-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700;
    color: #2c1500; line-height: 1.2; margin-bottom: 4px;
}
.page-divider {
    width: 50px; height: 2px;
    background: linear-gradient(90deg, #8B4513, transparent);
    margin: 10px 0 16px;
}
.page-body {
    font-family: 'Lato', sans-serif;
    font-size: 0.82rem; color: #3d2000; line-height: 1.7;
}

/* ── Streamlit widget theming ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label {
    font-family: 'Lato', sans-serif !important;
    font-size: 0.68rem !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    color: #5a2a00 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,245,220,0.95) !important;
    border: 1px solid rgba(139,69,19,0.3) !important;
    border-radius: 4px !important;
    font-family: 'Lato', sans-serif !important;
    color: #2c1500 !important;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: rgba(255,245,220,0.95) !important;
    border: 1px solid rgba(139,69,19,0.3) !important;
    border-radius: 4px !important;
    font-family: 'Lato', sans-serif !important;
    color: #2c1500 !important;
}
div[data-testid="stButton"] > button {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.95rem !important;
    background: linear-gradient(135deg, #8B4513, #5a2008) !important;
    color: #f5d78e !important;
    border: 1px solid #c87941 !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
}

/* ── Rec cards ── */
.rec-item {
    background: rgba(255,240,205,0.75);
    border-left: 3px solid #8B4513;
    padding: 8px 12px; margin-bottom: 8px;
    border-radius: 0 4px 4px 0;
}
.rec-rank  { font-family:'Playfair Display',serif; font-size:0.68rem; color:#8B4513; font-weight:700; }
.rec-title { font-family:'Playfair Display',serif; font-size:0.9rem; color:#2c1500; font-weight:700; line-height:1.25; }
.rec-meta  { font-family:'Lato',sans-serif; font-size:0.7rem; color:#6b3a00; margin-top:1px; }
.rec-reason {
    font-family:'Lato',sans-serif; font-size:0.72rem; color:#4a2800;
    font-style:italic; margin-top:3px; padding-top:3px;
    border-top:1px solid rgba(139,69,19,0.15);
}
.rec-scroll {
    max-height: calc(92vh - 200px);
    overflow-y: auto; padding-right: 4px;
}
.rec-scroll::-webkit-scrollbar { width: 3px; }
.rec-scroll::-webkit-scrollbar-thumb { background: rgba(139,69,19,0.25); border-radius:2px; }

/* Intro button override */
.intro-btn div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #8B4513, #5a2008) !important;
    font-size: 1.1rem !important;
    padding: 14px 32px !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Library background (always shown)
# =============================================================================
COLORS = [
    "#c0392b","#e74c3c","#8e44ad","#2980b9","#27ae60","#f39c12",
    "#d35400","#16a085","#2c3e50","#7f8c8d","#e67e22","#1abc9c",
    "#6c5ce7","#a29bfe","#fd79a8","#74b9ff","#55efc4","#fab1a0",
]

def make_shelf(n):
    spines = "".join(
        f'<div class="book-spine" style="height:{55+(i*17)%45}px;background:{COLORS[i%len(COLORS)]};"></div>'
        for i in range(n)
    )
    return f'<div class="shelf">{spines}</div>'

particles = "".join(
    f'<div class="particle" style="left:{(i*137)%100}%;width:{2+(i%2)}px;height:{2+(i%2)}px;'
    f'animation-duration:{6+(i*0.7)%8}s;animation-delay:{(i*1.3)%6}s;"></div>'
    for i in range(18)
)

st.markdown(f"""
<div class="library-bg">
    {make_shelf(38)}{make_shelf(36)}{make_shelf(40)}{make_shelf(35)}{make_shelf(42)}
    {particles}
</div>
<div class="glow-overlay"></div>
""", unsafe_allow_html=True)

# =============================================================================
# Data & model
# =============================================================================
@st.cache_data
def load_data():
    return pd.read_csv("Books.csv"), pd.read_csv("Ratings.csv")

@st.cache_resource
def train_model(_ratings):
    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(_ratings[["user_id","book_id","rating"]], reader)
    model   = KNNBasic(k=50, sim_options={"name":"pearson","user_based":True}, verbose=False)
    model.fit(data.build_full_trainset())
    return model

books, ratings = load_data()
title_of      = dict(zip(books["book_id"], books["title"]))
popular_books = set(ratings["book_id"].value_counts()[lambda x: x >= 20].index)
cf_model      = train_model(ratings)
all_users     = sorted(ratings["user_id"].unique())

def get_cf_recs(user_id, top_n=10):
    seen = set(ratings.loc[ratings["user_id"]==user_id,"book_id"])
    scored = [
        (title_of.get(b, str(b)), cf_model.predict(user_id, b).est)
        for b in books["book_id"] if b not in seen and b in popular_books
    ]
    return sorted(scored, key=lambda x: -x[1])[:top_n]

class RankedPick(BaseModel):
    title:  str = Field(description="Exact book title from the candidate list.")
    reason: str = Field(description="One sentence on why this fits the user's preference.")

def rerank(api_key, candidates, preference, top_n):
    client = genai.Client(api_key=api_key)
    cf_map = {t:s for t,s in candidates}
    meta   = books[books["title"].isin(cf_map)].drop_duplicates("title")
    rows   = [
        f"- {r['title']} by {r['authors']} "
        f"[{int(r['original_publication_year']) if pd.notna(r['original_publication_year']) else '?'}] "
        f"| Avg:{r['average_rating']} | CF:{cf_map.get(r['title'],0):.2f}"
        for _, r in meta.iterrows()
    ]
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"User preference: {preference}\n\nCandidates:\n" + "\n".join(rows),
        config={
            "system_instruction": (
                "You are a book concierge. Re-rank the candidate list to best match "
                "the user's preference. Use only titles from the list. "
                f"Return the top {top_n} picks best to worst."
            ),
            "response_mime_type": "application/json",
            "response_schema": list[RankedPick],
        },
    )
    return resp.parsed

# =============================================================================
# INTRO
# =============================================================================
if st.session_state.phase == "intro":
    st.markdown("""
    <div class="intro-screen">
        <div style="text-align:center;">
            <p class="subtitle">A Goodreads Project</p>
            <h1 class="main-title">The <em>Book</em><br>Recommender</h1>
        </div>
        <p style="font-family:'Lato',sans-serif;font-size:0.72rem;
           color:rgba(245,215,142,0.4);letter-spacing:0.18em;text-transform:uppercase;">
           Open to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1.5, 1, 1.5])
    with mid:
        st.markdown('<div class="intro-btn">', unsafe_allow_html=True)
        if st.button("📖  Start Recommendation  →", use_container_width=True):
            st.session_state.phase = "form"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# BOOK FORM + RESULTS
# =============================================================================
else:
    st.markdown('<div class="book-wrapper">', unsafe_allow_html=True)

    left_col, spine_col, right_col = st.columns([11, 0.3, 11])

    # ── LEFT PAGE ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
        <p class="page-chapter">Chapter I</p>
        <h2 class="page-heading">Your Reading Profile</h2>
        <div class="page-divider"></div>
        <p class="page-body" style="margin-bottom:18px;">
            Select your reader profile and preferences below.
            Our model will consult the shelves on your behalf.
        </p>
        """, unsafe_allow_html=True)

        selected_user = st.selectbox("Select your user ID", options=all_users, index=0)
        top_n         = st.slider("How many recommendations?", 5, 15, 10, 5)

        st.markdown("""
        <p class="page-chapter" style="margin-top:18px;">Chapter II — AI Re-ranking</p>
        <p class="page-body" style="margin-bottom:10px;">
            Describe your mood or genre and Gemini will re-rank your picks.
        </p>
        """, unsafe_allow_html=True)

        gemini_key = st.text_input("Gemini API key", type="password",
                                    placeholder="Paste key here (optional)")
        preference = st.text_area("Your preference / mood",
                                   value="I love gripping thrillers I can't put down",
                                   height=68)
        rerank_n   = st.slider("Re-ranked picks to show", 3, 8, 5)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        get_recs = st.button("✦  Find My Books", use_container_width=True)

    # ── SPINE ──────────────────────────────────────────────────────────────
    with spine_col:
        st.markdown("<div style='height:92vh'></div>", unsafe_allow_html=True)

    # ── RIGHT PAGE ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("""
        <p class="page-chapter">Your Recommendations</p>
        <h2 class="page-heading">Curated Picks</h2>
        <div class="page-divider" style="background:linear-gradient(90deg,transparent,#8B4513);"></div>
        """, unsafe_allow_html=True)

        if get_recs:
            with st.spinner("Consulting the shelves…"):
                st.session_state.cf_recs  = get_cf_recs(selected_user, top_n)
                st.session_state.reranked = []
                if gemini_key and preference.strip():
                    try:
                        st.session_state.reranked = rerank(
                            gemini_key, st.session_state.cf_recs, preference, rerank_n
                        )
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
            st.session_state.phase = "results"

        if st.session_state.cf_recs:
            picks  = st.session_state.reranked or []
            cf_list = st.session_state.cf_recs

            st.markdown('<div class="rec-scroll">', unsafe_allow_html=True)

            if picks:
                st.markdown('<p class="page-body" style="margin-bottom:10px;">✦ Re-ranked by Gemini:</p>',
                            unsafe_allow_html=True)
                for i, pick in enumerate(picks, 1):
                    row    = books[books["title"] == pick.title]
                    author = row["authors"].values[0] if not row.empty else ""
                    avg    = row["average_rating"].values[0] if not row.empty else ""
                    st.markdown(f"""
                    <div class="rec-item">
                        <div class="rec-rank">#{i}</div>
                        <div class="rec-title">{pick.title}</div>
                        <div class="rec-meta">{author}{"  ·  ★ "+str(avg) if avg else ""}</div>
                        <div class="rec-reason">{pick.reason}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<p class="page-body" style="margin-bottom:10px;">Top picks based on readers like you:</p>',
                            unsafe_allow_html=True)
                for i, (title, score) in enumerate(cf_list, 1):
                    row    = books[books["title"] == title]
                    author = row["authors"].values[0] if not row.empty else ""
                    st.markdown(f"""
                    <div class="rec-item">
                        <div class="rec-rank">#{i}</div>
                        <div class="rec-title">{title}</div>
                        <div class="rec-meta">{author}  ·  CF {score:.2f}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;">
                <p style="font-family:'Playfair Display',serif;font-size:2.2rem;opacity:0.12;">📖</p>
                <p class="page-body" style="opacity:0.4;margin-top:10px;">
                    Fill in the left page and<br>click Find My Books.
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("""
<div style="position:fixed;bottom:0;left:0;right:0;z-index:5;padding:6px;text-align:center;
     background:linear-gradient(transparent,rgba(0,0,0,0.3));">
    <span style="font-family:'Lato',sans-serif;font-size:0.6rem;
          color:rgba(245,215,142,0.3);letter-spacing:0.12em;">
        OPAN 6604 · Project 2 · Cliff Akins · UBCF Pearson k=50 · Gemini gemini-2.5-flash-lite
    </span>
</div>
""", unsafe_allow_html=True)
