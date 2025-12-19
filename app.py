import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Recommendation System Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🚀 Recommendation System – Managerial Insights Dashboard")

st.markdown("""
This interactive dashboard demonstrates **Recommendation System concepts** and provides  
**business-friendly statistical insights** from user-uploaded datasets.
""")

st.divider()

# ---------------- TOPICS SECTION ----------------
st.subheader("📌 Topics Explored (Conceptual Gist)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Matrix Factorization**  
    Decomposes large interaction matrices into latent features for recommendations.

    **Content-Based Filtering**  
    Recommends items similar to user preferences.
    """)

with col2:
    st.markdown("""
    **Collaborative Filtering**  
    Uses user–user or item–item similarities.

    **Cosine Similarity & Text Embedding**  
    Converts text/features into vectors for similarity scoring.
    """)

st.divider()

# ---------------- FILE UPLOADER ----------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---------------- OVERVIEW ----------------
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.divider()

    # ---------------- PREVIEW ----------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    st.divider()

    # ---------------- DESCRIPTIVE STATS ----------------
    st.subheader("📈 Descriptive Statistics (Manager View)")
    st.write("Quick summary to understand scale, spread and central tendency.")
    st.dataframe(df.describe())

    st.divider()

    # ---------------- NUMERIC DATA ----------------
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # ---------------- DISTRIBUTION PLOT ----------------
    st.subheader("📊 Distribution Analysis")

    if numeric_df.shape[1] > 0:
        col = st.selectbox("Select a numeric column", numeric_df.columns)

        fig, ax = plt.subplots()
        ax.hist(numeric_df[col], bins=20)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.divider()

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("🔥 Correlation Heatmap (Feature Relationships)")

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()

        fig, ax = plt.subplots()
        im = ax.imshow(corr)
        plt.colorbar(im)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    st.divider()

    # ---------------- TIME SERIES ----------------
    st.subheader("⏳ Time Series Insight (Moving Average)")

    if numeric_df.shape[1] > 0:
        ts_col = st.selectbox("Select column for trend analysis", numeric_df.columns)
        window = st.slider("Moving Average Window", 2, 10, 3)

        ma = numeric_df[ts_col].rolling(window).mean()

        fig, ax = plt.subplots()
        ax.plot(numeric_df[ts_col], label="Original")
        ax.plot(ma, label="Moving Average")
        ax.legend()
        ax.set_title("Trend Analysis")
        st.pyplot(fig)

    st.divider()

    # ---------------- 3D VISUALIZATION ----------------
    st.subheader("🌐 3D Visualization (Advanced Insight)")

    if numeric_df.shape[1] >= 3:
        x = numeric_df.columns[0]
        y = numeric_df.columns[1]
        z = numeric_df.columns[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            numeric_df[x],
            numeric_df[y],
            numeric_df[z],
            c=numeric_df[z],
            cmap="viridis"
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title("3D Feature Relationship")
        st.pyplot(fig)
    else:
        st.info("At least 3 numerical columns required for 3D plot.")

    st.divider()

    # ---------------- MANAGERIAL INSIGHTS ----------------
    st.subheader("🧠 Managerial Insights")

    st.markdown("""
    - Helps identify **important variables** influencing outcomes  
    - Reveals **patterns & trends** useful for recommendations  
    - Supports **data-driven decision making**  
    - Forms the foundation for **collaborative & content-based filtering**
    """)

else:
    st.info("👆 Upload a CSV file to generate insights.")
