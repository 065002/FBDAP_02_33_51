import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📘 Book Recommendation System", "📊 Insights & Concepts"]
)

# =====================================================
# PAGE 1: BOOK RECOMMENDATION SYSTEM
# =====================================================
if page == "📘 Book Recommendation System":

    st.title("📘 Book Recommendation System")

    st.write(
        "Enter a book name and the system will recommend similar books "
        "using **text similarity and cosine similarity**."
    )

    # -------- Identify Book Column --------
    book_col = df.columns[0]   # assumes first column = book name

    # -------- Text Vectorization --------
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df[book_col].astype(str))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # -------- User Input --------
    user_input = st.text_input("🔍 Enter a book name")

    if user_input:
        matches = get_close_matches(
            user_input,
            df[book_col].astype(str).tolist(),
            n=1,
            cutoff=0.4
        )

        if matches:
            selected_book = matches[0]
            idx = df[df[book_col] == selected_book].index[0]

            similarity_scores = list(enumerate(cosine_sim[idx]))
            similarity_scores = sorted(
                similarity_scores,
                key=lambda x: x[1],
                reverse=True
            )

            recommendations = similarity_scores[1:6]
            rec_books = [df.iloc[i[0]][book_col] for i in recommendations]

            st.success(f"Recommendations similar to **{selected_book}**")
            for book in rec_books:
                st.write("📖", book)

        else:
            st.warning("No similar book found. Try another name.")

# =====================================================
# PAGE 2: INSIGHTS & CONCEPTS
# =====================================================
else:

    st.title("📊 Dataset Insights & Recommendation Concepts")

    # -------- Dataset Overview --------
    st.subheader("📂 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # -------- Preview --------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.divider()

    # -------- Concepts Section --------
    st.subheader("📌 Topics Explored")

    st.markdown("""
    **Matrix Factorization**  
    Decomposes user–item interaction matrices into latent features.

    **Content-Based Filtering**  
    Recommends books similar in content to user preferences.

    **Collaborative Filtering**  
    Uses user behavior similarity to recommend items.

    **Cosine Similarity**  
    Measures similarity between book feature vectors.

    **Text Embedding**  
    Converts book titles/descriptions into numerical vectors.
    """)

    st.divider()

    # -------- Statistical Insights --------
    st.subheader("📈 Statistical Insights")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        st.dataframe(numeric_df.describe())
    else:
        st.info("No numerical columns available for statistical analysis.")

    st.divider()

    # -------- Linear Algebra Explanation --------
    st.subheader("🧮 Linear Algebra in Recommendation Systems")

    st.write("""
    - User–Item data is represented as matrices  
    - Similarity is computed using vector operations  
    - Dimensionality reduction helps uncover hidden patterns
    """)

    st.divider()

    # -------- Time Series (Conceptual) --------
    st.subheader("⏳ Time Series (Conceptual)")

    st.write("""
    Moving Averages help smooth trends and understand changes in user behavior over time.
    """)

# ---------------- FOOTER ----------------
st.caption("Built using Streamlit | Book Recommendation System")
