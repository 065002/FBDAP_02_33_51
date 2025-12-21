import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")
st.write(
    "Type a book name (full or partial) and get similar book recommendations."
)

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Book Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------- CHANGE COLUMN NAME HERE IF REQUIRED --------
    BOOK_COLUMN = "Book-Title"   # 👈 change ONLY this if your column name is different

    if BOOK_COLUMN not in df.columns:
        st.error(
            f"Column '{BOOK_COLUMN}' not found in dataset. "
            "Please check your book title column name."
        )
    else:
        df = df[[BOOK_COLUMN]].dropna()
        df[BOOK_COLUMN] = df[BOOK_COLUMN].astype(str)

        # ---------------- USER INPUT ----------------
        user_input = st.text_input(
            "Enter a book name (partial name also works):"
        )

        if user_input:
            # ---------------- VECTORIZE BOOK TITLES ----------------
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df[BOOK_COLUMN])

            # ---------------- USER QUERY VECTOR ----------------
            user_vector = tfidf.transform([user_input])

            # ---------------- COSINE SIMILARITY ----------------
            similarity_scores = cosine_similarity(
                user_vector, tfidf_matrix
            ).flatten()

            df["Similarity"] = similarity_scores

            # ---------------- FILTER & SORT ----------------
            recommendations = (
                df[df["Similarity"] > 0]
                .sort_values(by="Similarity", ascending=False)
                .head(10)
            )

            # ---------------- DISPLAY RESULTS ----------------
            if recommendations.empty:
                st.warning("No similar books found. Try another keyword.")
            else:
                st.subheader("📖 Recommended Books")
                for i, row in recommendations.iterrows():
                    st.write(f"• {row[BOOK_COLUMN]}")

else:
    st.info("👆 Upload a CSV file to get book recommendations.")
