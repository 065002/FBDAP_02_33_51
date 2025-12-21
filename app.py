import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")
st.caption("Find similar books even if you remember only part of the name")

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Book Dataset (CSV)",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully")

    # ---------------- COLUMN SELECTION ----------------
    st.subheader("📌 Select Book Title Column")

    book_col = st.selectbox(
        "Choose the column that contains book names",
        df.columns
    )

    # Clean data
    df = df[[book_col]].dropna()
    df[book_col] = df[book_col].astype(str)

    # ---------------- USER INPUT ----------------
    st.divider()
    st.subheader("🔍 Enter a Book Name")

    user_input = st.text_input(
        "Type a book name (full or partial)",
        placeholder="e.g. harry, alchemist, atomic"
    )

    if user_input:

        # ---------- FUZZY MATCHING ----------
        matches = difflib.get_close_matches(
            user_input,
            df[book_col].tolist(),
            n=1,
            cutoff=0.3
        )

        if not matches:
            st.error("No similar book found. Try a different keyword.")
        else:
            selected_book = matches[0]

            st.info(f"Showing recommendations for: **{selected_book}**")

            # ---------- TF-IDF VECTORIZATION ----------
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df[book_col])

            # Index of selected book
            book_index = df[df[book_col] == selected_book].index[0]

            # ---------- COSINE SIMILARITY ----------
            similarity_scores = cosine_similarity(
                tfidf_matrix[book_index],
                tfidf_matrix
            ).flatten()

            # Top recommendations
            similar_indices = similarity_scores.argsort()[-6:-1][::-1]

            recommendations = df.iloc[similar_indices][book_col]

            # ---------- DISPLAY RESULTS ----------
            st.divider()
            st.subheader("📖 Recommended Books")

            for book in recommendations:
                st.write("•", book)

else:
    st.info("👆 Upload a CSV file to start recommending books.")
