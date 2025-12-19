import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Recommendation System Dashboard", layout="wide")

# ===================== TITLE =====================
st.title("📊 Recommendation System – Analytics & Decision Support Dashboard")

st.write("""
This dashboard demonstrates **recommendation system concepts** and 
**managerial insights** using uploaded business datasets.
""")

st.divider()

# ===================== CONCEPTUAL OVERVIEW =====================
st.subheader("📌 Recommendation System Concepts (Gist)")

st.markdown("""
- **Matrix Factorization**: Decomposes large user–item matrices to uncover latent patterns.  
- **Content-Based Filtering**: Recommends items similar to user preferences.  
- **Collaborative Filtering**: Uses user behavior similarities for recommendations.  
- **Cosine Similarity**: Measures similarity between users or items.  
- **Text Embedding**: Converts text into numerical vectors for similarity analysis.
""")

st.divider()

# ===================== FILE UPLOAD =====================
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ===================== OVERVIEW =====================
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # ===================== PREVIEW =====================
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ===================== NUMERIC DATA =====================
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    st.divider()

    # ===================== DESCRIPTIVE STATS =====================
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(numeric_df.describe())

    st.divider()

    # ===================== CORRELATION HEATMAP =====================
    if numeric_df.shape[1] > 1:
        st.subheader("🔥 Correlation Heatmap (Managerial Insight)")
        fig, ax = plt.subplots()
        cax = ax.matshow(numeric_df.corr())
        fig.colorbar(cax)
        ax.set_title("Correlation Between Numerical Variables")
        st.pyplot(fig)

    st.divider()

    # ===================== MOVING AVERAGE =====================
    st.subheader("⏳ Time Series Trend (Moving Average)")

    if numeric_df.shape[1] > 0:
        col = st.selectbox("Select Column", numeric_df.columns)
        window = st.slider("Window Size", 2, 10, 3)

        df["MA"] = numeric_df[col].rolling(window).mean()

        fig, ax = plt.subplots()
        ax.plot(numeric_df[col], label="Original")
        ax.plot(df["MA"], label="Moving Avg")
        ax.legend()
        st.pyplot(fig)

    st.divider()

    # ===================== REGRESSION =====================
    st.subheader("📐 Regression Insight (Inferential Statistics)")

    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0]
        y = numeric_df.iloc[:, 1]
        coef = np.polyfit(x, y, 1)
        st.write(f"**Regression Equation:** y = {coef[0]:.2f}x + {coef[1]:.2f}")

    st.divider()

    # ===================== COSINE SIMILARITY =====================
    st.subheader("🧭 Cosine Similarity (Recommendation Logic)")

    if numeric_df.shape[1] > 1:
        sim = cosine_similarity(numeric_df.fillna(0))
        st.write("Sample Similarity Matrix:")
        st.dataframe(sim[:5, :5])

    st.divider()

    # ===================== MATRIX FACTORIZATION =====================
    st.subheader("🧮 Matrix Factorization (SVD – Conceptual Demo)")

    if numeric_df.shape[1] > 1:
        svd = TruncatedSVD(n_components=2)
        reduced = svd.fit_transform(numeric_df.fillna(0))
        st.write("Latent Factors (Sample):")
        st.dataframe(reduced[:5])

    st.divider()

    # ===================== 3D VISUALIZATION =====================
    st.subheader("🧊 3D Visualization (Latent Space View)")

    if numeric_df.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            numeric_df.iloc[:, 0],
            numeric_df.iloc[:, 1],
            numeric_df.iloc[:, 2]
        )
        ax.set_xlabel(numeric_df.columns[0])
        ax.set_ylabel(numeric_df.columns[1])
        ax.set_zlabel(numeric_df.columns[2])
        st.pyplot(fig)

else:
    st.info("👆 Upload a dataset to activate the dashboard.")
