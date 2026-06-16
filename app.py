"""
Goodreads Recommender — 3-phase UX
Intro → Form (Ch1 left, Ch2 right) → Results (picks left, model info right)
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
if "selected_user" not in st.session_state:
    st.session_state.selected_user = None
if "preference" not in st.session_state:
    st.session_state.preference = ""

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

.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0rem !important; }

/* ── Library background ── */
.library-bg {
    position: fixed; inset: 0;
    background: linear-gradient(160deg, #1a0a00 0%, #3d1a00 30%, #5c2d00 55%, #2a0d00 100%);
    overflow: hidden; z-index: 0;
}
.shelf {
    position: absolute; left: 0; right: 0; height: 110px;
    display: flex; align-items: flex-end; gap: 3px; padding: 0 20px;
    border-bottom: 8px solid #3a1f00;
    animation: shelfFloat 8s ease-in-out infinite;
}
.shelf:nth-child(1){bottom:0px;    animation-delay:0s;}
.shelf:nth-child(2){bottom:115px;  animation-delay:1s;}
.shelf:nth-child(3){bottom:230px;  animation-delay:2s;}
.shelf:nth-child(4){bottom:345px;  animation-delay:0.5s;}
.shelf:nth-child(5){bottom:460px;  animation-delay:1.5s;}
@keyframes shelfFloat{0%,100%{transform:translateY(0);}50%{transform:translateY(-4px);}}
.book-spine{width:28px;border-radius:3px 3px 0 0;box-shadow:inset -3px 0 6px rgba(0,0,0,0.4);}
.particle{position:absolute;border-radius:50%;background:rgba(255,210,100,0.5);animation:drift linear infinite;}
@keyframes drift{0%{transform:translateY(100vh);opacity:0;}10%{opacity:1;}90%{opacity:0.4;}100%{transform:translateY(-20px) translateX(20px);opacity:0;}}
.glow-overlay{position:fixed;inset:0;background:radial-gradient(ellipse at 50% 60%,rgba(255,160,50,0.07) 0%,transparent 70%);pointer-events:none;z-index:1;animation:flicker 4s ease-in-out infinite;}
@keyframes flicker{0%,100%{opacity:1;}50%{opacity:0.75;}}

/* ── Intro title ── */
.intro-title-block {
    position: relative;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 20vh;
    padding-bottom: 4vh;
    text-align: center;
}
.title-backdrop {
    background: rgba(20, 8, 0, 0.62);
    border-radius: 16px;
    padding: 28px 56px 32px;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(245,215,142,0.12);
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 700;
    color: #f5d78e;
    line-height: 1.15;
    animation: fadeUp 1.8s ease forwards;
}
.main-title em { font-style: italic; color: #ffd97a; }
.subtitle {
    font-family: 'Lato', sans-serif; font-size: 0.9rem;
    font-weight: 300; color: rgba(245,215,142,0.75);
    letter-spacing: 0.28em; text-transform: uppercase;
    animation: fadeUp 2.2s ease forwards;
    margin-bottom: 10px;
}
@keyframes fadeUp{0%{opacity:0;transform:translateY(-20px);}100%{opacity:1;transform:translateY(0);}}

/* ── Intro button: fully transparent columns ── */
.intro-btn-row {
    position: relative;
    z-index: 20;
    display: flex;
    justify-content: center;
    margin-top: 0;
}
.intro-btn-row [data-testid="stVerticalBlock"] {
    background: transparent !important;
    box-shadow: none !important;
    min-height: unset !important;
    padding: 4px 8px !important;
    border-radius: 0 !important;
}
.intro-btn-row div[data-testid="stButton"] > button {
    font-size: 1.05rem !important;
    padding: 13px 36px !important;
}

/* ── Book pages: ONLY inside .book-wrapper ── */
.book-wrapper {
    position: relative;
    z-index: 10;
    padding: 16px 12px;
}
.book-wrapper [data-testid="stHorizontalBlock"] > div:nth-child(1) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(160deg, #fdf6e3 0%, #f9edd8 50%, #f5e6cc 100%) !important;
    min-height: 92vh !important;
    padding: 36px 32px 28px !important;
    border-radius: 6px 0 0 6px !important;
    box-shadow: -4px 0 24px rgba(0,0,0,0.5), inset -6px 0 16px rgba(0,0,0,0.06) !important;
    position: relative !important; z-index: 10 !important;
}
.book-wrapper [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(180deg,#5a2a00,#3d1500,#5a2a00) !important;
    min-height: 92vh !important;
    box-shadow: -5px 0 14px rgba(0,0,0,0.5),5px 0 14px rgba(0,0,0,0.5) !important;
    z-index: 11 !important;
}
.book-wrapper [data-testid="stHorizontalBlock"] > div:nth-child(3) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(200deg, #fdf6e3 0%, #f9edd8 50%, #f5e6cc 100%) !important;
    min-height: 92vh !important;
    padding: 36px 32px 28px !important;
    border-radius: 0 6px 6px 0 !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5), inset 6px 0 16px rgba(0,0,0,0.06) !important;
    position: relative !important; z-index: 10 !important;
}

/* ── Page typography ── */
.page-chapter{font-family:'Lato',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.25em;text-transform:uppercase;color:#8B4513;margin-bottom:6px;}
.page-heading{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:#2c1500;line-height:1.2;margin-bottom:4px;}
.page-divider{width:50px;height:2px;background:linear-gradient(90deg,#8B4513,transparent);margin:10px 0 16px;}
.page-divider-right{width:50px;height:2px;background:linear-gradient(90deg,transparent,#8B4513);margin:10px 0 16px;}
.page-body{font-family:'Lato',sans-serif;font-size:0.82rem;color:#3d2000;line-height:1.7;}

/* ── Widget theming ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label {
    font-family:'Lato',sans-serif !important;font-size:0.68rem !important;
    font-weight:700 !important;letter-spacing:0.14em !important;
    text-transform:uppercase !important;color:#5a2a00 !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background:rgba(255,245,220,0.95) !important;
    border:1px solid rgba(139,69,19,0.3) !important;
    border-radius:4px !important;
    font-family:'Lato',sans-serif !important;color:#2c1500 !important;
}
div[data-testid="stButton"] > button {
    font-family:'Playfair Display',serif !important;
    font-size:0.95rem !important;
    background:linear-gradient(135deg,#8B4513,#5a2008) !important;
    color:#f5d78e !important;border:1px solid #c87941 !important;
    border-radius:6px !important;width:100% !important;padding:10px !important;
    box-shadow:0 4px 12px rgba(0,0,0,0.3) !important;
    transition:all 0.3s !important;
}
div[data-testid="stButton"] > button:hover{transform:translateY(-2px) !important;}

/* ── Rec cards ── */
.rec-item{background:rgba(255,240,205,0.75);border-left:3px solid #8B4513;padding:9px 13px;margin-bottom:8px;border-radius:0 4px 4px 0;}
.rec-rank{font-family:'Playfair Display',serif;font-size:0.68rem;color:#8B4513;font-weight:700;}
.rec-title{font-family:'Playfair Display',serif;font-size:0.9rem;color:#2c1500;font-weight:700;line-height:1.25;}
.rec-meta{font-family:'Lato',sans-serif;font-size:0.7rem;color:#6b3a00;margin-top:1px;}
.rec-reason{font-family:'Lato',sans-serif;font-size:0.72rem;color:#4a2800;font-style:italic;margin-top:3px;padding-top:3px;border-top:1px solid rgba(139,69,19,0.15);}

/* ── Model info cards ── */
.info-card{background:rgba(255,240,205,0.6);border:1px solid rgba(139,69,19,0.2);padding:14px 16px;margin-bottom:12px;border-radius:4px;}
.info-card-title{font-family:'Playfair Display',serif;font-size:0.88rem;font-weight:700;color:#2c1500;margin-bottom:4px;}
.info-card-body{font-family:'Lato',sans-serif;font-size:0.78rem;color:#4a2800;line-height:1.65;}
.metric-row{display:flex;gap:12px;margin-bottom:12px;}
.metric-box{flex:1;background:rgba(139,69,19,0.08);border:1px solid rgba(139,69,19,0.18);border-radius:4px;padding:10px 12px;text-align:center;}
.metric-val{font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:#8B4513;}
.metric-lbl{font-family:'Lato',sans-serif;font-size:0.62rem;color:#6b3a00;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;}

.rec-scroll{max-height:calc(92vh - 190px);overflow-y:auto;padding-right:4px;}
.rec-scroll::-webkit-scrollbar{width:3px;}
.rec-scroll::-webkit-scrollbar-thumb{background:rgba(139,69,19,0.25);border-radius:2px;}
.page-number{font-family:'Playfair Display',serif;font-size:0.72rem;color:rgba(139,69,19,0.4);margin-top:16px;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Library background
# =============================================================================
COLORS = [
    "#c0392b","#e74c3c","#8e44ad","#2980b9","#27ae60","#f39c12",
    "#d35400","#16a085","#2c3e50","#7f8c8d","#e67e22","#1abc9c",
    "#6c5ce7","#a29bfe","#fd79a8","#74b9ff","#55efc4","#fab1a0",
]
def make_shelf(n):
    s = "".join(
        f'<div class="book-spine" style="height:{55+(i*17)%45}px;background:{COLORS[i%len(COLORS)]};"></div>'
        for i in range(n))
    return f'<div class="shelf">{s}</div>'

particles = "".join(
    f'<div class="particle" style="left:{(i*137)%100}%;width:{2+(i%2)}px;height:{2+(i%2)}px;'
    f'animation-duration:{6+(i*0.7)%8}s;animation-delay:{(i*1.3)%6}s;"></div>'
    for i in range(18))

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
    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(_ratings[["user_id","book_id","rating"]], reader)
    model  = KNNBasic(k=50, sim_options={"name":"pearson","user_based":True}, verbose=False)
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
# PHASE: INTRO
# =============================================================================
if st.session_state.phase == "intro":

    # Title with dark backdrop pill — readable over any shelf
    st.markdown("""
    <div class="intro-title-block">
        <div class="title-backdrop">
            <p class="subtitle">A Goodreads Project</p>
            <h1 class="main-title">The <em>Book</em><br>Recommender</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Button in its own transparent wrapper — NO columns
    st.markdown('<div class="intro-btn-row">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1.2, 2])
    with col2:
        if st.button("📖  Start Recommendation  →", use_container_width=True):
            st.session_state.phase = "form"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PHASE: FORM  (Ch1 left | Ch2 right)
# =============================================================================
elif st.session_state.phase == "form":
    st.markdown('<div class="book-wrapper">', unsafe_allow_html=True)
    left_col, spine_col, right_col = st.columns([11, 0.3, 11])

    with left_col:
        st.markdown("""
        <p class="page-chapter">Chapter I</p>
        <h2 class="page-heading">Your Reading Profile</h2>
        <div class="page-divider"></div>
        <p class="page-body" style="margin-bottom:20px;">
            Choose your reader ID and how many recommendations you'd like.
            Our collaborative filtering model will find readers just like you
            and surface the books they loved that you haven't read yet.
        </p>
        """, unsafe_allow_html=True)

        selected_user = st.selectbox("Select your user ID", options=all_users, index=0)
        top_n         = st.slider("How many recommendations?", 5, 15, 10, 5)

        st.markdown("""
        <p class="page-body" style="margin-top:24px;opacity:0.5;font-style:italic;">
            Complete Chapter II on the right, then click <strong>Find My Books</strong>.
        </p>
        <p class="page-number">— 1 —</p>
        """, unsafe_allow_html=True)

    with spine_col:
        st.markdown("<div style='height:92vh'></div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <p class="page-chapter">Chapter II</p>
        <h2 class="page-heading">AI Re-ranking</h2>
        <div class="page-divider-right"></div>
        <p class="page-body" style="margin-bottom:18px;">
            Optionally describe your current mood or favourite genre.
            Gemini will re-rank your collaborative filtering candidates
            to match your stated preference, with a reason for each pick.
        </p>
        """, unsafe_allow_html=True)

        gemini_key = st.text_input("Gemini API key", type="password",
                                    placeholder="Paste key here (optional)")
        preference = st.text_area("Your preference / mood",
                                   value="I love gripping thrillers I can't put down",
                                   height=90)
        rerank_n   = st.slider("Re-ranked picks to show", 3, 8, 5)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if st.button("✦  Find My Books", use_container_width=True):
            with st.spinner("Consulting the shelves…"):
                st.session_state.cf_recs       = get_cf_recs(selected_user, top_n)
                st.session_state.reranked      = []
                st.session_state.selected_user = selected_user
                st.session_state.preference    = preference
                if gemini_key and preference.strip():
                    try:
                        st.session_state.reranked = rerank(
                            gemini_key, st.session_state.cf_recs, preference, rerank_n
                        )
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
            st.session_state.phase = "results"
            st.rerun()

        st.markdown('<p class="page-number" style="text-align:right;">— 2 —</p>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PHASE: RESULTS  (picks left | model info right)
# =============================================================================
elif st.session_state.phase == "results":
    st.markdown('<div class="book-wrapper">', unsafe_allow_html=True)
    left_col, spine_col, right_col = st.columns([11, 0.3, 11])

    picks    = st.session_state.reranked
    cf_list  = st.session_state.cf_recs
    user     = st.session_state.selected_user
    pref     = st.session_state.preference
    n_rated  = len(ratings[ratings["user_id"] == user]) if user else 0
    n_users  = ratings["user_id"].nunique()
    n_books  = ratings["book_id"].nunique()

    with left_col:
        st.markdown(f"""
        <p class="page-chapter">Your Recommendations</p>
        <h2 class="page-heading">Curated Picks</h2>
        <div class="page-divider"></div>
        <p class="page-body" style="margin-bottom:12px;">
            {"✦ Re-ranked by Gemini for: <em>\"" + pref + "\"</em>" if picks
             else "Top picks based on readers like you:"}
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="rec-scroll">', unsafe_allow_html=True)

        if picks:
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
            for i, (title, score) in enumerate(cf_list, 1):
                row    = books[books["title"] == title]
                author = row["authors"].values[0] if not row.empty else ""
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-rank">#{i}</div>
                    <div class="rec-title">{title}</div>
                    <div class="rec-meta">{author}  ·  CF score {score:.2f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<p class="page-number">— 3 —</p>', unsafe_allow_html=True)

    with spine_col:
        st.markdown("<div style='height:92vh'></div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <p class="page-chapter">About This Recommendation</p>
        <h2 class="page-heading">How It Works</h2>
        <div class="page-divider-right"></div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{n_rated:,}</div>
                <div class="metric-lbl">Books You've Rated</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{n_users:,}</div>
                <div class="metric-lbl">Total Readers</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{n_books:,}</div>
                <div class="metric-lbl">Books in Catalog</div>
            </div>
        </div>

        <div class="info-card">
            <div class="info-card-title">Collaborative Filtering (UBCF)</div>
            <div class="info-card-body">
                Your recommendations are generated by User-Based Collaborative
                Filtering using Pearson similarity with k=50 neighbours.
                The model finds the 50 readers most similar to you based on
                shared rating patterns, then predicts how much you'd enjoy
                books they loved that you haven't read yet.
            </div>
        </div>

        <div class="info-card">
            <div class="info-card-title">Model Performance</div>
            <div class="info-card-body">
                <strong>RMSE:</strong> 1.03 &nbsp;·&nbsp;
                <strong>Precision@10:</strong> 0.660 &nbsp;·&nbsp;
                <strong>Recall@10:</strong> 0.794<br><br>
                UBCF Pearson k=50 was selected from a four-model bake-off
                (Baseline, UBCF cosine k=10, UBCF Pearson k=50, IBCF cosine k=50).
                It achieves the best ranking quality even though the Baseline
                has a lower RMSE — illustrating that rating accuracy and
                recommendation quality don't always agree.
            </div>
        </div>

        <div class="info-card">
            <div class="info-card-title">AI Re-ranking Layer</div>
            <div class="info-card-body">
                When a preference is provided, Google Gemini
                (gemini-2.5-flash-lite) re-ranks the CF candidates using
                book metadata — title, author, year, and average rating —
                to surface the best match for your stated mood or genre.
                The LLM reorders existing CF picks; it never generates
                new titles from scratch.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("← Start Over", use_container_width=True):
            st.session_state.phase    = "form"
            st.session_state.cf_recs  = []
            st.session_state.reranked = []
            st.rerun()

        st.markdown('<p class="page-number" style="text-align:right;">— 4 —</p>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("""
<div style="position:fixed;bottom:0;left:0;right:0;z-index:5;padding:6px;
     text-align:center;background:linear-gradient(transparent,rgba(0,0,0,0.3));">
    <span style="font-family:'Lato',sans-serif;font-size:0.6rem;
          color:rgba(245,215,142,0.3);letter-spacing:0.12em;">
        OPAN 6604 · Project 2 · Cliff Akins · UBCF Pearson k=50 · Gemini gemini-2.5-flash-lite
    </span>
</div>
""", unsafe_allow_html=True)
