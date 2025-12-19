import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Recommendation System Dashboard",
    layout="wide"
)

# --------------------------------------------------
# TITLE & INTRO
# --------------------------------------------------
st.title("📊 Recommendation System – Managerial Analytics Dashboard")

st.write("""
This deployed dashboard supports **data-driven decision making** by providing
quick insights, visual analytics, and conceptual grounding for
**Recommendation Systems**.
""")

st.divider()

# --------------------------------------------------
# TOPICS EXPLORED (GIST – AS REQUIRED)
# --------------------------------------------------
st.subheader("📌 Topics Explored (Conceptual Gist)")

st.markdown("""
- **Matrix Factorization:** Decomposes large interaction matrices to uncover hidden user–item patterns.  
- **Content-Based Filtering:** Recommends items based on similarity in item attributes.  
- **Collaborative Filtering:** Uses user behavior similarity to generate recommendations.  
- **Cosine Similarity:** Measures similarity between users or items in vector space.  
- **Text Embedding:** Converts textual information into numerical form for similarity comparison.
""")

st.divider()

# --------------------------------------------------
# FILE UPLOAD (GENERAL & SAFE)
# --------------------------------------------------
st.subheader("📂 Upload Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload any CSV dataset to generate insights",
    type="csv"
)

if uploaded_file is not None:

    # READ DATA SAFELY
    df = pd.read_csv(uploaded_file)

    # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------
    st.subheader("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10))

    st.divider()

    # --------------------------------------------------
    # DESCRIPTIVE STATISTICS (MANAGER FRIENDLY)
    # --------------------------------------------------
    st.subheader("📈 Descriptive Statistical Insights")

    st.write("""
Helps managers quickly understand:
- Central tendency (mean, median)
- Spread and variability
- Overall data behavior
""")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        st.dataframe(numeric_df.describe())
    else:
        st.info("No numerical columns available for statistical summary.")

    st.divider()

    # --------------------------------------------------
    # CORRELATION HEATMAP (KEY FOR INSIGHTS)
    # --------------------------------------------------
    st.subheader("🔥 Correlation Analysis (Managerial Insight)")

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()

        fig, ax = plt.subplots()
        im = ax.imshow(corr)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        fig.colorbar(im)
        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)
    else:
        st.info("Correlation requires at least two numerical variables.")

    st.divider()

    # --------------------------------------------------
    # TIME SERIES – MOVING AVERAGE
    # --------------------------------------------------
    st.subheader("⏳ Time Series Insight (Moving Average)")

    if not numeric_df.empty:
        col = st.selectbox("Select column for trend analysis", numeric_df.columns)
        window = st.slider("Moving Average Window", 2, 10, 3)

        ma = numeric_df[col].rolling(window).mean()

        fig, ax = plt.subplots()
        ax.plot(numeric_df[col], label="Original")
        ax.plot(ma, label="Moving Average")
        ax.legend()
        ax.set_title("Trend Smoothing using Moving Average")

        st.pyplot(fig)
    else:
        st.info("Time series analysis requires numerical data.")

    st.divider()

    # --------------------------------------------------
    # 3D VISUALIZATION (ADVANCED + IMPRESSIVE)
    # --------------------------------------------------
    st.subheader("🧊 3D Data Visualization (Advanced Insight)")

    if numeric_df.shape[1] >= 3:
        x = st.selectbox("X-axis", numeric_df.columns, index=0)
        y = st.selectbox("Y-axis", numeric_df.columns, index=1)
        z = st.selectbox("Z-axis", numeric_df.columns, index=2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

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
        ax.set_title("3D Scatter Plot – Multivariate Insight")

        st.pyplot(fig)
    else:
        st.info("3D visualization requires at least three numerical columns.")

else:
    st.info("👆 Upload a CSV file to start analysis.")
