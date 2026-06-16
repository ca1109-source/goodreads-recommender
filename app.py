"""
Project 2 — Goodreads Recommender App
--------------------------------------
Run from the same folder as Books.csv and Ratings.csv:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field

# ── Surprise ──────────────────────────────────────────────────────────────────
from surprise import KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split

# ── Gemini ────────────────────────────────────────────────────────────────────
from google import genai

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Goodreads Recommender",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Goodreads Book Recommender")
st.caption(
    "Collaborative filtering (UBCF Pearson k=50) + optional Gemini re-ranking."
)

# =============================================================================
# Load data — cached so it only runs once
# =============================================================================
@st.cache_data
def load_data():
    books   = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    return books, ratings

books, ratings = load_data()
title_of = dict(zip(books["book_id"], books["title"]))

# =============================================================================
# Train model — cached so it only runs once
# =============================================================================
@st.cache_resource
def train_model(ratings):
    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(
                   ratings[["user_id", "book_id", "rating"]], reader)
    full_ts  = data.build_full_trainset()
    model    = KNNBasic(
                   k=50,
                   sim_options={"name": "pearson", "user_based": True},
                   verbose=False)
    model.fit(full_ts)
    return model

with st.spinner("Training UBCF model on full dataset…"):
    cf_model = train_model(ratings)

# Popular books filter (same threshold as notebook)
MIN_RATINGS   = 20
counts        = ratings["book_id"].value_counts()
popular_books = set(counts[counts >= MIN_RATINGS].index)

# =============================================================================
# CF recommendation helper
# =============================================================================
def get_cf_recommendations(user_id, top_n=10):
    seen   = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    scored = [
        (title_of.get(b, f"book_id={b}"), cf_model.predict(user_id, b).est)
        for b in books["book_id"]
        if b not in seen and b in popular_books
    ]
    return sorted(scored, key=lambda x: -x[1])[:top_n]

# =============================================================================
# Gemini re-ranking helper
# =============================================================================
class RankedPick(BaseModel):
    title:  str = Field(description="Exact book title from the candidate list.")
    reason: str = Field(description="One sentence on why this fits the user's preference.")

def rerank_with_gemini(api_key, candidates, candidate_meta, preference, top_n=5):
    client = genai.Client(api_key=api_key)
    MODEL  = "gemini-2.5-flash-lite"

    cf_score_map = {title: score for title, score in candidates}

    catalog_rows = []
    for _, row in candidate_meta.iterrows():
        cf_score = cf_score_map.get(row["title"], "N/A")
        year     = int(row["original_publication_year"]) \
                   if pd.notna(row["original_publication_year"]) else "Unknown"
        catalog_rows.append(
            f"- {row['title']} by {row['authors']} "
            f"[{year}] | Avg rating: {row['average_rating']} | "
            f"CF score: {cf_score:.2f}"
        )
    catalog = "\n".join(catalog_rows)

    system_instruction = (
        "You are a book concierge. You are given a list of book candidates "
        "pre-selected by a collaborative filtering model based on this user's "
        "rating history. Re-rank them to best match the user's stated preference. "
        "Use only books from the provided list — do not invent new titles. "
        f"Return the top {top_n} picks in order from best to worst fit."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            f"User preference: {preference}\n\n"
            f"Candidate books:\n{catalog}"
        ),
        config={
            "system_instruction":  system_instruction,
            "response_mime_type":  "application/json",
            "response_schema":     list[RankedPick],
        },
    )
    return response.parsed

# =============================================================================
# Sidebar — user selection & settings
# =============================================================================
st.sidebar.header("Settings")

all_users   = sorted(ratings["user_id"].unique())
selected_user = st.sidebar.selectbox(
    "Select a user",
    options=all_users,
    index=0,
    help="Choose a user ID to generate recommendations for."
)

top_n = st.sidebar.slider(
    "Number of CF recommendations",
    min_value=5, max_value=20, value=10, step=5
)

st.sidebar.divider()
st.sidebar.subheader("Gemini Re-ranking (optional)")

gemini_key  = st.sidebar.text_input(
    "Gemini API key",
    type="password",
    help="Paste your Gemini API key. It is never stored or logged."
)
preference  = st.sidebar.text_area(
    "Your preference / mood",
    value="I love gripping thrillers and page-turners I can't put down",
    height=80
)
rerank_n    = st.sidebar.slider(
    "Number of re-ranked picks",
    min_value=3, max_value=10, value=5, step=1
)
run_rerank  = st.sidebar.button("✨ Re-rank with Gemini", type="primary")

# =============================================================================
# Main panel — user stats
# =============================================================================
user_ratings = ratings[ratings["user_id"] == selected_user]
n_rated      = len(user_ratings)
avg_given    = user_ratings["rating"].mean()

c1, c2 = st.columns(2)
c1.metric("User ID",        selected_user)
c2.metric("Books rated",    f"{n_rated:,}")

st.divider()

# =============================================================================
# CF Recommendations
# =============================================================================
st.subheader(f"🤖 Top-{top_n} Collaborative Filtering Picks")
st.caption("Model: UBCF · Pearson similarity · k=50")

with st.spinner("Generating CF recommendations…"):
    cf_recs = get_cf_recommendations(selected_user, top_n=top_n)

# Enrich with metadata
cf_titles   = [t for t, _ in cf_recs]
cf_scores   = [s for _, s in cf_recs]
cf_meta     = (
    books[books["title"].isin(cf_titles)][
        ["title", "authors", "original_publication_year", "average_rating"]
    ]
    .drop_duplicates(subset="title")
)

cf_display = pd.DataFrame({
    "Rank":         range(1, len(cf_recs) + 1),
    "Title":        cf_titles,
    "CF Score":     [round(s, 2) for s in cf_scores],
}).merge(cf_meta, on="Title", how="left")

cf_display = cf_display.rename(columns={
    "authors":                    "Author(s)",
    "original_publication_year":  "Year",
    "average_rating":             "Avg Rating",
})
cf_display["Year"] = cf_display["Year"].apply(
    lambda y: int(y) if pd.notna(y) else "—"
)

st.dataframe(
    cf_display[["Rank", "Title", "Author(s)", "Year", "Avg Rating", "CF Score"]],
    use_container_width=True,
    hide_index=True,
)

# =============================================================================
# Gemini Re-ranking
# =============================================================================
st.divider()
st.subheader("✨ Gemini AI Re-ranking")

if run_rerank:
    if not gemini_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
    elif not preference.strip():
        st.warning("Please enter a preference or mood in the sidebar.")
    else:
        # Build candidate metadata for the LLM
        candidate_ids = [
            books.loc[books["title"] == title, "book_id"].values[0]
            for title, _ in cf_recs
            if not books.loc[books["title"] == title, "book_id"].empty
        ]
        candidate_meta = (
            books[books["book_id"].isin(candidate_ids)][
                ["book_id", "title", "authors",
                 "original_publication_year", "average_rating"]
            ]
            .drop_duplicates(subset="book_id")
        )

        with st.spinner("Asking Gemini to re-rank…"):
            try:
                reranked = rerank_with_gemini(
                    gemini_key, cf_recs, candidate_meta,
                    preference, top_n=rerank_n
                )

                st.success(f"Re-ranked for preference: *\"{preference}\"*")

                for i, pick in enumerate(reranked, 1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {pick.title}**")
                        st.caption(pick.reason)

                # Side-by-side comparison
                st.divider()
                st.subheader("📊 CF vs. Gemini Re-rank Comparison")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**CF Top 10**")
                    st.dataframe(
                        pd.DataFrame({
                            "CF Rank": range(1, len(cf_titles[:10]) + 1),
                            "Title":   cf_titles[:10],
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col2:
                    st.markdown(f"**Gemini Top {rerank_n}**")
                    st.dataframe(
                        pd.DataFrame({
                            "LLM Rank": range(1, len(reranked) + 1),
                            "Title":    [p.title  for p in reranked],
                            "Reason":   [p.reason for p in reranked],
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as e:
                st.error(f"Gemini API error: {e}")
else:
    st.info(
        "Enter your Gemini API key and a preference in the sidebar, "
        "then click **✨ Re-rank with Gemini**."
    )

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    "Model: UBCF Pearson k=50 (scikit-surprise) · "
    "LLM re-ranking: Google Gemini gemini-2.5-flash-lite · "
    "Built for OPAN 6604 Project 2 · Cliff Akins"
)
