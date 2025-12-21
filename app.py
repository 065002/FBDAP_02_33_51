import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Recommendation System", layout="wide")

# =====================================================
# TITLE
# =====================================================
st.title("📚 Recommendation System – Decision Support Dashboard")

st.write("""
This application builds a **generic recommendation system** using  
**Collaborative Filtering, Cosine Similarity, and Matrix Factorization**.
Users can upload **any dataset** and generate recommendations dynamically.
""")

st.divider()

# =====================================================
# CONCEPT OVERVIEW
# =====================================================
st.subheader("📌 Concepts Used (Gist)")

st.markdown("""
- **Collaborative Filtering** – Learns from user–item interactions  
- **Cosine Similarity** – Measures similarity between users/items  
- **Matrix Factorization (SVD)** – Extracts latent features  
- **Linear Algebra** – Matrix operations on interaction data  
- **Statistics** – Descriptive stats & similarity inference  
""")

st.divider()

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =====================================================
    # DATA PREVIEW
    # =====================================================
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # =====================================================
    # COLUMN SELECTION (THIS PREVENTS KEYERROR)
    # =====================================================
    st.subheader("⚙️ Select Columns for Recommendation")

    user_col = st.selectbox("Select USER column", df.columns)
    item_col = st.selectbox("Select ITEM column (Book/Product)", df.columns)
    rating_col = st.selectbox("Select RATING / SCORE column", df.columns)

    st.divider()

    # =====================================================
    # DATA PREPARATION
    # =====================================================
    df_model = df[[user_col, item_col, rating_col]].dropna()

    st.success("Columns selected successfully. Data prepared.")

    # =====================================================
    # USER–ITEM MATRIX
    # =====================================================
    user_item_matrix = df_model.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        fill_value=0
    )

    st.subheader("🧮 User–Item Interaction Matrix")
    st.dataframe(user_item_matrix.head())

    st.divider()

    # =====================================================
    # COSINE SIMILARITY (ITEM-BASED)
    # =====================================================
    st.subheader("🧭 Cosine Similarity (Item-Based)")

    item_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    st.dataframe(similarity_df.iloc[:5, :5])

    st.divider()

    # =====================================================
    # MATRIX FACTORIZATION (SVD)
    # =====================================================
    st.subheader("🧠 Matrix Factorization (Latent Features)")

    svd = TruncatedSVD(n_components=5)
    latent_matrix = svd.fit_transform(user_item_matrix)

    latent_df = pd.DataFrame(latent_matrix, index=user_item_matrix.index)
    st.dataframe(latent_df.head())

    st.divider()

    # =====================================================
    # RECOMMENDATION ENGINE
    # =====================================================
    st.subheader("🎯 Get Book Recommendation")

    selected_item = st.selectbox(
        "Select a Book / Item",
        user_item_matrix.columns
    )

    if st.button("Recommend Similar Items"):
        scores = similarity_df[selected_item].sort_values(ascending=False)
        recommendations = scores[1:6]  # Top 5 excluding itself

        st.success("Top Recommended Items:")
        st.dataframe(recommendations)

else:
    st.info("👆 Upload a CSV file to start building the recommendation system.")
