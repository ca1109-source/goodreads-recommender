"""
Goodreads Recommender — Redesigned UX
Animated library intro → book flip → open book form experience
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field
import os

from surprise import KNNBasic, Dataset, Reader
from google import genai

st.set_page_config(
    page_title="Goodreads Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "phase" not in st.session_state:
    st.session_state.phase = "intro"
if "cf_recs" not in st.session_state:
    st.session_state.cf_recs = []
if "reranked" not in st.session_state:
    st.session_state.reranked = []

# =============================================================================
# Global CSS + animations
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

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
.shelf:nth-child(1)  { bottom: 0px;   animation-delay: 0s; }
.shelf:nth-child(2)  { bottom: 115px; animation-delay: 1s; }
.shelf:nth-child(3)  { bottom: 230px; animation-delay: 2s; }
.shelf:nth-child(4)  { bottom: 345px; animation-delay: 0.5s; }
.shelf:nth-child(5)  { bottom: 460px; animation-delay: 1.5s; }

@keyframes shelfFloat {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-4px); }
}

.book-spine {
    width: 28px;
    border-radius: 3px 3px 0 0;
    position: relative;
    box-shadow: inset -3px 0 6px rgba(0,0,0,0.4), 2px 0 4px rgba(0,0,0,0.3);
}

.particle {
    position: absolute;
    width: 3px; height: 3px;
    border-radius: 50%;
    background: rgba(255,210,100,0.6);
    animation: drift linear infinite;
}
@keyframes drift {
    0%   { transform: translateY(100vh) translateX(0);    opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.5; }
    100% { transform: translateY(-20px) translateX(30px); opacity: 0; }
}

.glow-overlay {
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at 50% 60%, rgba(255,160,50,0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 1;
    animation: flicker 4s ease-in-out infinite;
}
@keyframes flicker {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.7; }
}

/* ── Intro screen ── */
.intro-screen {
    position: relative;
    z-index: 10;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 40px;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5.5rem);
    font-weight: 700;
    color: #f5d78e;
    text-align: center;
    text-shadow: 0 0 40px rgba(255,180,50,0.6), 0 0 80px rgba(255,140,20,0.3), 2px 4px 8px rgba(0,0,0,0.8);
    letter-spacing: 0.04em;
    line-height: 1.15;
    animation: titleAppear 1.8s ease forwards;
}
.main-title em { font-style: italic; color: #ffeaa0; }

@keyframes titleAppear {
    0%   { opacity: 0; transform: translateY(-30px); }
    100% { opacity: 1; transform: translateY(0); }
}

.subtitle {
    font-family: 'Lato', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: rgba(245,215,142,0.7);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    animation: titleAppear 2.2s ease forwards;
}

/* ── Open book layout ── */
.book-container {
    position: relative;
    z-index: 10;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
}

.open-book {
    position: relative;
    width: min(960px, 96vw);
    height: 88vh;
    max-height: 700px;
    display: grid;
    grid-template-columns: 1fr 10px 1fr;
    animation: bookOpen 0.6s ease forwards;
    filter: drop-shadow(0 30px 60px rgba(0,0,0,0.8));
    z-index: 20;
}

@keyframes bookOpen {
    0%   { opacity: 0; transform: scale(0.85) rotateX(8deg); }
    100% { opacity: 1; transform: scale(1) rotateX(0deg); }
}

.book-page {
    background: linear-gradient(160deg, #fdf6e3 0%, #f9edd8 40%, #f5e6cc 100%);
    padding: 32px 28px 24px;
    position: relative;
    isolation: isolate;
    z-index: 20;
    overflow: hidden;
}
.book-page::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        transparent, transparent 27px,
        rgba(180,120,40,0.07) 27px, rgba(180,120,40,0.07) 28px
    );
    pointer-events: none;
    z-index: 0;
}
.book-page-left  {
    border-radius: 4px 0 0 4px;
    box-shadow: -4px 0 20px rgba(0,0,0,0.4), inset -8px 0 16px rgba(0,0,0,0.06);
}
.book-page-right {
    border-radius: 0 4px 4px 0;
    box-shadow: 4px 0 20px rgba(0,0,0,0.4), inset 8px 0 16px rgba(0,0,0,0.06);
}

.book-spine-center {
    background: linear-gradient(180deg, #5a2a00, #3d1500, #5a2a00);
    box-shadow: -4px 0 12px rgba(0,0,0,0.5), 4px 0 12px rgba(0,0,0,0.5);
    z-index: 25;
}

.page-chapter {
    font-family: 'Lato', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #8B4513;
    margin-bottom: 8px;
    position: relative;
    z-index: 1;
}
.page-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #2c1500;
    margin-bottom: 4px;
    line-height: 1.2;
    position: relative;
    z-index: 1;
}
.page-divider {
    width: 50px;
    height: 2px;
    background: linear-gradient(90deg, #8B4513, transparent);
    margin: 10px 0 14px;
}
.page-body {
    font-family: 'Lato', sans-serif;
    font-size: 0.82rem;
    color: #3d2000;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

/* Override Streamlit widget styles for book theme */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label {
    font-family: 'Lato', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a2a00 !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: rgba(255,245,220,0.9) !important;
    border: 1px solid rgba(139,69,19,0.35) !important;
    border-radius: 4px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.82rem !important;
    color: #2c1500 !important;
}
div[data-testid="stButton"] button {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.95rem !important;
    background: linear-gradient(135deg, #8B4513, #5a2008) !important;
    color: #f5d78e !important;
    border: 1px solid #c87941 !important;
    border-radius: 6px !important;
    padding: 8px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    width: 100% !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
}

/* Rec cards */
.rec-item {
    background: rgba(255,240,210,0.7);
    border-left: 3px solid #8B4513;
    padding: 8px 12px;
    margin-bottom: 7px;
    border-radius: 0 4px 4px 0;
    position: relative;
    z-index: 1;
}
.rec-rank  { font-family:'Playfair Display',serif; font-size:0.68rem; color:#8B4513; font-weight:700; }
.rec-title { font-family:'Playfair Display',serif; font-size:0.88rem; color:#2c1500; font-weight:700; line-height:1.25; }
.rec-meta  { font-family:'Lato',sans-serif; font-size:0.7rem; color:#6b3a00; margin-top:1px; }
.rec-reason {
    font-family:'Lato',sans-serif; font-size:0.72rem; color:#4a2800;
    font-style:italic; margin-top:3px; padding-top:3px;
    border-top:1px solid rgba(139,69,19,0.15);
}

.page-number {
    position: absolute;
    bottom: 14px;
    font-family: 'Playfair Display', serif;
    font-size: 0.75rem;
    color: rgba(139,69,19,0.45);
    z-index: 1;
}
.page-number-left  { left: 28px; }
.page-number-right { right: 28px; }

.right-page-scroll {
    max-height: calc(88vh - 200px);
    overflow-y: auto;
    padding-right: 6px;
    position: relative;
    z-index: 1;
}
.right-page-scroll::-webkit-scrollbar { width: 3px; }
.right-page-scroll::-webkit-scrollbar-track { background: transparent; }
.right-page-scroll::-webkit-scrollbar-thumb { background: rgba(139,69,19,0.25); border-radius: 2px; }

/* Tighten up Streamlit spacing inside book */
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stSlider { margin-bottom: 0 !important; padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Animated library background
# =============================================================================
COLORS = [
    "#c0392b","#e74c3c","#8e44ad","#2980b9","#27ae60",
    "#f39c12","#d35400","#16a085","#2c3e50","#7f8c8d",
    "#e67e22","#1abc9c","#6c5ce7","#a29bfe","#fd79a8",
    "#74b9ff","#55efc4","#fab1a0","#b2bec3","#dfe6e9",
]

def make_shelf(num_books):
    books_html = ""
    for i in range(num_books):
        h   = 55 + (i * 17) % 45
        col = COLORS[i % len(COLORS)]
        books_html += f'<div class="book-spine" style="height:{h}px;background:{col};"></div>'
    return f'<div class="shelf">{books_html}</div>'

particles_html = "".join([
    f'<div class="particle" style="left:{(i*137)%100}%;'
    f'animation-duration:{6+(i*0.7)%8}s;'
    f'animation-delay:{(i*1.3)%6}s;'
    f'width:{2+(i%2)}px;height:{2+(i%2)}px;"></div>'
    for i in range(18)
])

st.markdown(f"""
<div class="library-bg">
    {make_shelf(38)}{make_shelf(35)}{make_shelf(40)}{make_shelf(36)}{make_shelf(42)}
    {particles_html}
</div>
<div class="glow-overlay"></div>
""", unsafe_allow_html=True)

# =============================================================================
# Data & model
# =============================================================================
@st.cache_data
def load_data():
    books   = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    return books, ratings

@st.cache_resource
def train_model(_ratings):
    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(_ratings[["user_id","book_id","rating"]], reader)
    full_ts = data.build_full_trainset()
    model   = KNNBasic(k=50, sim_options={"name":"pearson","user_based":True}, verbose=False)
    model.fit(full_ts)
    return model

books, ratings = load_data()
title_of      = dict(zip(books["book_id"], books["title"]))
MIN_RATINGS   = 20
counts        = ratings["book_id"].value_counts()
popular_books = set(counts[counts >= MIN_RATINGS].index)

with st.spinner("Loading recommender…"):
    cf_model = train_model(ratings)

all_users = sorted(ratings["user_id"].unique())

# =============================================================================
# Helpers
# =============================================================================
def get_cf_recs(user_id, top_n=10):
    seen   = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    scored = [
        (title_of.get(b, f"book_id={b}"), cf_model.predict(user_id, b).est)
        for b in books["book_id"]
        if b not in seen and b in popular_books
    ]
    return sorted(scored, key=lambda x: -x[1])[:top_n]

class RankedPick(BaseModel):
    title:  str = Field(description="Exact book title from the candidate list.")
    reason: str = Field(description="One sentence on why this fits the user's preference.")

def rerank(api_key, candidates, preference, top_n=5):
    client = genai.Client(api_key=api_key)
    cf_map = {t: s for t, s in candidates}
    meta   = books[books["title"].isin([t for t,_ in candidates])].drop_duplicates("title")
    rows   = []
    for _, r in meta.iterrows():
        yr = int(r["original_publication_year"]) if pd.notna(r["original_publication_year"]) else "?"
        rows.append(
            f"- {r['title']} by {r['authors']} [{yr}] | "
            f"Avg: {r['average_rating']} | CF: {cf_map.get(r['title'],0):.2f}"
        )
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
            "response_schema":    list[RankedPick],
        },
    )
    return resp.parsed

# =============================================================================
# INTRO PHASE
# =============================================================================
if st.session_state.phase == "intro":
    st.markdown("""
    <div class="intro-screen">
        <div style="text-align:center;">
            <p class="subtitle">A Goodreads Project</p>
            <h1 class="main-title">The <em>Book</em><br>Recommender</h1>
        </div>
        <p style="font-family:'Lato',sans-serif;font-size:0.75rem;
           color:rgba(245,215,142,0.45);text-align:center;
           letter-spacing:0.18em;text-transform:uppercase;">
           Open to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        if st.button("📖  Start Recommendation  →", use_container_width=True):
            st.session_state.phase = "form"
            st.rerun()

# =============================================================================
# FORM + RESULTS PHASE
# =============================================================================
elif st.session_state.phase in ("form", "results"):

    st.markdown('<div class="book-container"><div class="open-book">', unsafe_allow_html=True)

    left_col, spine_col, right_col = st.columns([10, 0.18, 10])

    # ── LEFT PAGE ─────────────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
        <div class="book-page book-page-left" style="height:88vh;max-height:700px;">
            <p class="page-chapter">Chapter I</p>
            <h2 class="page-heading">Your Reading Profile</h2>
            <div class="page-divider"></div>
            <p class="page-body" style="margin-bottom:14px;">
                Select your reader profile and preferences below.
                Our model will consult the shelves on your behalf.
            </p>
        </div>
        """, unsafe_allow_html=True)

        selected_user = st.selectbox("Select your user ID", options=all_users, index=0)
        top_n         = st.slider("How many recommendations?", 5, 15, 10, 5)

        st.markdown("""
        <div style="position:relative;z-index:1;margin-top:10px;">
            <p class="page-chapter">Chapter II — AI Re-ranking</p>
            <p class="page-body" style="margin-bottom:8px;">
                Describe your mood or genre and Gemini will re-rank your picks.
            </p>
        </div>
        """, unsafe_allow_html=True)

        gemini_key = st.text_input("Gemini API key", type="password", placeholder="Paste key here (optional)")
        preference = st.text_area("Your preference / mood",
                                   value="I love gripping thrillers I can't put down",
                                   height=65)
        rerank_n   = st.slider("Re-ranked picks to show", 3, 8, 5, 1)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        get_recs = st.button("✦  Find My Books", use_container_width=True)

        st.markdown('<span class="page-number page-number-left">1</span>', unsafe_allow_html=True)

    # ── SPINE ─────────────────────────────────────────────────────────────────
    with spine_col:
        st.markdown("""
        <div class="book-spine-center" style="height:88vh;max-height:700px;"></div>
        """, unsafe_allow_html=True)

    # ── RIGHT PAGE ────────────────────────────────────────────────────────────
    with right_col:
        st.markdown("""
        <div class="book-page book-page-right" style="height:88vh;max-height:700px;">
            <p class="page-chapter">Your Recommendations</p>
            <h2 class="page-heading">Curated Picks</h2>
            <div class="page-divider" style="background:linear-gradient(90deg,transparent,#8B4513);"></div>
        </div>
        """, unsafe_allow_html=True)

        # Trigger recommendations
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

        # Display results
        if st.session_state.cf_recs:
            display_list = st.session_state.reranked
            cf_list      = st.session_state.cf_recs

            st.markdown('<div class="right-page-scroll">', unsafe_allow_html=True)

            if display_list:
                st.markdown("""
                <p class="page-body" style="margin-bottom:10px;position:relative;z-index:1;">
                ✦ Re-ranked by Gemini for your preference:</p>
                """, unsafe_allow_html=True)
                for i, pick in enumerate(display_list, 1):
                    row    = books[books["title"] == pick.title]
                    author = row["authors"].values[0] if not row.empty else ""
                    avg    = row["average_rating"].values[0] if not row.empty else ""
                    st.markdown(f"""
                    <div class="rec-item">
                        <div class="rec-rank">#{i}</div>
                        <div class="rec-title">{pick.title}</div>
                        <div class="rec-meta">{author}{"  ·  ★ " + str(avg) if avg else ""}</div>
                        <div class="rec-reason">{pick.reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p class="page-body" style="margin-bottom:10px;position:relative;z-index:1;">
                Top picks based on readers like you:</p>
                """, unsafe_allow_html=True)
                for i, (title, score) in enumerate(cf_list, 1):
                    row    = books[books["title"] == title]
                    author = row["authors"].values[0] if not row.empty else ""
                    st.markdown(f"""
                    <div class="rec-item">
                        <div class="rec-rank">#{i}</div>
                        <div class="rec-title">{title}</div>
                        <div class="rec-meta">{author}  ·  CF score {score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px;position:relative;z-index:1;">
                <p style="font-family:'Playfair Display',serif;font-size:2.5rem;opacity:0.15;">📖</p>
                <p class="page-body" style="opacity:0.45;margin-top:10px;">
                    Fill in the left page and<br>click Find My Books.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="page-number page-number-right">2</span>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("""
<div style="position:fixed;bottom:0;left:0;right:0;z-index:5;padding:6px;text-align:center;
     background:linear-gradient(transparent,rgba(0,0,0,0.35));">
    <span style="font-family:'Lato',sans-serif;font-size:0.6rem;
          color:rgba(245,215,142,0.3);letter-spacing:0.15em;">
        OPAN 6604 · Project 2 · Cliff Akins · UBCF Pearson k=50 · Gemini gemini-2.5-flash-lite
    </span>
</div>
""", unsafe_allow_html=True)
