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
    "Enter a book name (full or partial). "
    "The system will recommend similar books using text similarity."
)

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Book Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset safely
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully")

    # ---------------- COLUMN SELECTION ----------------
    st.subheader("Select Book Title Column")

    book_column = st.selectbox(
        "Choose the column that contains book names",
        df.columns
    )

    # Clean book names
    df = df[[book_column]].dropna()
    df[book_column] = df[book_column].astype(str)

    st.divider()

    # ---------------- USER INPUT ----------------
    st.subheader("Enter Book Name")

    user_input = st.text_input(
        "Type a book name (partial names also work)",
        placeholder="e.g. Harry, Atomic, Rich, Data..."
    )

    # ---------------- RECOMMENDATION LOGIC ----------------
    if user_input:

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df[book_column])

        # Transform user input
        user_vector = vectorizer.transform([user_input])

        # Cosine similarity
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Get top matches
        df["similarity"] = similarity_scores
        recommendations = df.sort_values(
            by="similarity",
            ascending=False
        )

        recommendations = recommendations[
            recommendations["similarity"] > 0
        ].head(10)

        st.divider()

        # ---------------- OUTPUT ----------------
        st.subheader("📖 Recommended Books")

        if len(recommendations) == 0:
            st.warning("No similar books found. Try another keyword.")
        else:
            for i, book in enumerate(recommendations[book_column], start=1):
                st.write(f"{i}. {book}")

else:
    st.info("👆 Upload a CSV file to start")
