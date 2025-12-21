import streamlit as st
import pandas as pd
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Book Recommendation System", layout="wide")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

st.title("📚 Book Recommendation System")

# ===================== TABS =====================
tab1, tab2 = st.tabs(["📖 Recommendation System", "📊 Dataset Insights"])

# ==================================================
# TAB 1 — BOOK RECOMMENDATION SYSTEM
# ==================================================
with tab1:
    st.subheader("🔍 Get Book Recommendations")

    st.write(
        "Select a column, enter a value, and get **relevant book recommendations** "
        "based strictly on the dataset."
    )

    # Column selection
    search_column = st.selectbox(
        "Select the column to search by",
        df.columns
    )

    user_input = st.text_input(
        f"Enter {search_column} value"
    )

    if user_input:
        # Convert column to string safely
        df[search_column] = df[search_column].astype(str)

        # Find close matches (NO random results)
        matches = get_close_matches(
            user_input,
            df[search_column].unique(),
            n=5,
            cutoff=0.6
        )

        if len(matches) == 0:
            st.warning("No close matches found in the dataset.")
        else:
            st.success("Matching entries found in dataset")

            matched_df = df[df[search_column].isin(matches)]

            # CONTENT-BASED RECOMMENDATION (TEXT)
            text_data = matched_df.apply(
                lambda row: " ".join(row.astype(str)),
                axis=1
            )

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(text_data)

            similarity = cosine_similarity(tfidf_matrix)

            st.subheader("📚 Recommended Books")

            recommended_indices = similarity[0].argsort()[::-1][1:6]

            recommendations = matched_df.iloc[recommended_indices]

            st.dataframe(recommendations.reset_index(drop=True))

# ==================================================
# TAB 2 — DATASET INSIGHTS
# ==================================================
with tab2:
    st.subheader("📊 Dataset Insights & Concepts")

    # Dataset overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("📌 Recommendation System Concepts (Gist)")

    st.markdown("""
    **Matrix Factorization**  
    Breaks user–item interaction matrices into latent features.

    **Content-Based Filtering**  
    Recommends items similar to what the user searched for.

    **Collaborative Filtering**  
    Uses user–user or item–item similarity.

    **Cosine Similarity**  
    Measures similarity between vectors.

    **Text Embedding**  
    Converts textual book data into numerical form for similarity comparison.
    """)

    st.divider()

    st.subheader("📈 Statistical Snapshot")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe())
    else:
        st.info("No numerical columns available for statistics.")

    st.divider()

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())
