import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

st.title("📚 Book Recommendation System")

# --------------------------------------------------
# LOAD DATA (PRE-UPLOADED DATASET)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigate",
    ["Book Recommendation System", "Dataset Insights"]
)

# ==================================================
# 1️⃣ BOOK RECOMMENDATION SYSTEM
# ==================================================
if menu == "Book Recommendation System":

    st.subheader("🔍 Find Book Recommendations")

    # Select column to search
    search_column = st.selectbox(
        "Select search type",
        df.columns
    )

    # User input (partial match allowed)
    user_input = st.text_input(
        f"Enter {search_column} (partial text allowed)"
    )

    # Number of recommendations
    top_n = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=50,
        value=5
    )

    # Optional rating filter
    rating_col = None
    rating_cols = [c for c in df.columns if "rating" in c.lower()]

    if rating_cols:
        rating_col = st.selectbox(
            "Select rating column (optional)",
            ["None"] + rating_cols
        )

    min_rating = None
    if rating_col and rating_col != "None":
        min_rating = st.slider(
            "Minimum rating",
            float(df[rating_col].min()),
            float(df[rating_col].max()),
            float(df[rating_col].min())
        )

    # --------------------------------------------------
    # RECOMMENDATION LOGIC
    # --------------------------------------------------
    if user_input:

        filtered_df = df[
            df[search_column]
            .astype(str)
            .str.contains(user_input, case=False, na=False)
        ]

        if rating_col and rating_col != "None":
            filtered_df = filtered_df[
                filtered_df[rating_col] >= min_rating
            ]

        if filtered_df.empty:
            st.error("❌ No matching books found.")
        else:
            st.success(f"✅ {len(filtered_df)} matching books found")

            st.dataframe(
                filtered_df.head(top_n).reset_index(drop=True)
            )

# ==================================================
# 2️⃣ DATASET INSIGHTS
# ==================================================
if menu == "Dataset Insights":

    st.subheader("📊 Dataset Insights")

    # Overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("📌 Topics Explored (Conceptual Gist)")
    st.markdown("""
    **Matrix Factorization** – Decomposing user–item matrices to uncover latent patterns  
    **Content-Based Filtering** – Recommending books based on similar attributes  
    **Collaborative Filtering** – Recommendations using user behavior similarities  
    **Cosine Similarity** – Measuring similarity between books/users  
    **Text Embedding** – Converting text (titles/authors) into numeric vectors  
    """)

    st.divider()

    st.subheader("📈 Descriptive Statistics")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe())
    else:
        st.info("No numerical columns available.")

    st.divider()

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))
