import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Book Recommendation System", layout="wide")

# ======================================================
# TITLE
# ======================================================
st.title("📚 Book Recommendation System – Decision Support Dashboard")

st.write("""
This application demonstrates a **Recommendation System** using  
**Collaborative Filtering, Matrix Factorization, and Cosine Similarity**  
to help users discover relevant books based on similarity patterns.
""")

st.divider()

# ======================================================
# CONCEPT OVERVIEW
# ======================================================
st.subheader("📌 Concepts Used (Gist)")

st.markdown("""
- **Collaborative Filtering**: Uses user–item interaction patterns  
- **Matrix Factorization (SVD)**: Extracts latent preferences  
- **Cosine Similarity**: Measures similarity between books  
- **Linear Algebra**: Matrix operations on user–item matrix  
- **Statistics**: Mean ratings & trends for decision insights
""")

st.divider()

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("📂 Upload Book Ratings Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # ======================================================
    # DATA PREPARATION
    # ======================================================
    st.subheader("⚙️ Data Preparation")

    # Rename columns here if needed
    user_col = "User_ID"
    item_col = "Book_Title"
    rating_col = "Rating"

    df = df[[user_col, item_col, rating_col]].dropna()

    st.success("User–Item–Rating structure identified successfully")

    # User–Item Matrix
    user_item_matrix = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        fill_value=0
    )

    st.write("User–Item Matrix Shape:", user_item_matrix.shape)

    st.divider()

    # ======================================================
    # MATRIX FACTORIZATION (SVD)
    # ======================================================
    st.subheader("🧮 Matrix Factorization (Latent Preferences)")

    svd = TruncatedSVD(n_components=10)
    latent_matrix = svd.fit_transform(user_item_matrix)

    st.write("Latent Feature Matrix (Sample):")
    st.dataframe(latent_matrix[:5])

    st.divider()

    # ======================================================
    # COSINE SIMILARITY (ITEM-ITEM)
    # ======================================================
    st.subheader("🧭 Book Similarity using Cosine Similarity")

    item_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    st.write("Sample Similarity Matrix:")
    st.dataframe(similarity_df.iloc[:5, :5])

    st.divider()

    # ======================================================
    # BOOK RECOMMENDATION ENGINE
    # ======================================================
    st.subheader("🎯 Get Book Recommendations")

    book_input = st.selectbox(
        "Select a book you like:",
        user_item_matrix.columns
    )

    if book_input:
        similarity_scores = similarity_df[book_input].sort_values(ascending=False)

        recommended_books = similarity_scores.iloc[1:6]

        st.success("Recommended Books Based on Similarity:")

        for book, score in recommended_books.items():
            st.write(f"📘 **{book}**  (Similarity Score: {score:.2f})")

    st.divider()

    # ======================================================
    # MANAGERIAL INSIGHTS
    # ======================================================
    st.subheader("📈 Managerial Insights")

    avg_ratings = df.groupby(item_col)[rating_col].mean().sort_values(ascending=False)

    st.write("Top Rated Books (Average Rating):")
    st.dataframe(avg_ratings.head(10))

    fig, ax = plt.subplots()
    avg_ratings.head(10).plot(kind="bar", ax=ax)
    ax.set_title("Top Rated Books")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.divider()

    # ======================================================
    # 3D LATENT SPACE VISUALIZATION
    # ======================================================
    st.subheader("🧊 3D Visualization – Latent Space (Conceptual)")

    if latent_matrix.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            latent_matrix[:, 0],
            latent_matrix[:, 1],
            latent_matrix[:, 2],
            alpha=0.6
        )
        ax.set_xlabel("Latent Feature 1")
        ax.set_ylabel("Latent Feature 2")
        ax.set_zlabel("Latent Feature 3")
        st.pyplot(fig)

else:
    st.info("👆 Upload a book ratings dataset to activate recommendations.")
