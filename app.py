import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

# -------------------------------------------------
# TITLE & INTRO
# -------------------------------------------------
st.title("📚 Book Recommendation System")

st.write(
    "Enter a book name (full or partial) and get recommendations "
    "for similar books based on content similarity."
)

st.divider()

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("FBDAP Dataset.csv")
    df = df.dropna(subset=["Book-Title"])
    return df

df = load_data()

# -------------------------------------------------
# TEXT VECTORIZATION
# -------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["Book-Title"])

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
book_input = st.text_input(
    "🔎 Enter a book name (partial name also works)",
    placeholder="e.g. Harry, Data, Python, Management..."
)

# -------------------------------------------------
# RECOMMENDATION LOGIC
# -------------------------------------------------
def recommend_books(user_input, top_n=5):
    matches = df[df["Book-Title"].str.contains(user_input, case=False)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]

    return df.iloc[similar_indices]["Book-Title"].values

# -------------------------------------------------
# SHOW RECOMMENDATIONS
# -------------------------------------------------
if book_input:
    recommendations = recommend_books(book_input)

    if recommendations is None:
        st.error("❌ No similar books found. Try a different keyword.")
    else:
        st.subheader("📖 Recommended Books for You")
        for i, book in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {book}**")
