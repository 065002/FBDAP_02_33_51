import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recommendation System Dashboard", layout="wide")

# ===================== TITLE =====================
st.title("📊 Recommendation System – Managerial Insights Dashboard")

st.write(
    "This dashboard converts a recommendation system project into an interactive decision-support tool. "
    "It helps managers quickly understand data quality, patterns, trends, and relationships relevant "
    "for recommendation systems."
)

st.divider()

# ===================== TOPICS EXPLORED =====================
st.subheader("📌 Recommendation System Concepts Covered")

st.markdown("""
- **Matrix Factorization:** Identifying hidden patterns in user–item interaction data  
- **Content-Based Filtering:** Recommendations using item attributes  
- **Collaborative Filtering:** Recommendations based on user behavior similarity  
- **Cosine Similarity:** Measuring similarity between users or items  
- **Text Embedding:** Converting textual information into numerical form for similarity analysis
""")

st.divider()

# ===================== FILE UPLOAD =====================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # ===================== EXECUTIVE OVERVIEW =====================
    st.subheader("📌 Executive Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", df.shape[0])
    c2.metric("Total Variables", df.shape[1])
    c3.metric("Numeric Features", numeric_df.shape[1])

    st.divider()

    # ===================== DATA QUALITY =====================
    st.subheader("🧹 Data Quality Check")

    missing_count = df.isnull().sum().sum()
    st.write(f"**Total Missing Values:** {missing_count}")

    if missing_count == 0:
        st.success("Dataset is clean with no missing values.")
    else:
        st.warning("Dataset contains missing values. Cleaning may be required.")

    st.divider()

    # ===================== DATA PREVIEW =====================
    st.subheader("🔍 Dataset Snapshot")
    st.dataframe(df.head(8))

    st.divider()

    # ===================== DESCRIPTIVE STATISTICS =====================
    st.subheader("📈 Statistical Summary (Managerial View)")
    st.write(
        "This section provides central tendency and variability measures "
        "to understand overall data behavior."
    )
    st.dataframe(df.describe())

    st.divider()

    # ===================== DISTRIBUTION ANALYSIS =====================
    st.subheader("📊 Distribution Analysis")

    if numeric_df.shape[1] > 0:
        col = st.selectbox("Select numeric column for distribution", numeric_df.columns)

        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    st.divider()

    # ===================== TREND ANALYSIS =====================
    st.subheader("⏳ Trend Analysis (Moving Average)")

    if numeric_df.shape[1] > 0:
        ts_col = st.selectbox("Select column for trend analysis", numeric_df.columns)
        window = st.slider("Moving Average Window", 2, 10, 3)

        df["Moving_Avg"] = df[ts_col].rolling(window=window).mean()

        fig, ax = plt.subplots()
        ax.plot(df[ts_col], label="Original Data")
        ax.plot(df["Moving_Avg"], label="Moving Average")
        ax.set_title("Trend Analysis")
        ax.legend()

        st.pyplot(fig)

    st.divider()

    # ===================== RELATIONSHIP ANALYSIS =====================
    st.subheader("🔗 Relationship Analysis (For Recommendations)")

    if numeric_df.shape[1] >= 2:
        x_col = st.selectbox("Select X-axis", numeric_df.columns)
        y_col = st.selectbox("Select Y-axis", numeric_df.columns, index=1)

        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Relationship Between Variables")

        st.pyplot(fig)

        coeff = np.corrcoef(df[x_col], df[y_col])[0, 1]
        st.write(f"**Correlation (Similarity Indicator):** {coeff:.2f}")

    st.divider()

    # ===================== CORRELATION HEATMAP =====================
    st.subheader("🧠 Similarity Structure (Correlation Matrix)")

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        fig, ax = plt.subplots()
        cax = ax.imshow(corr)
        ax.set_title("Correlation Heatmap")
        plt.colorbar(cax)

        st.pyplot(fig)

    st.divider()

    # ===================== MANAGERIAL INSIGHTS =====================
    st.subheader("📌 Managerial Insights")

    st.markdown("""
    - High correlations indicate potential similarity useful for collaborative filtering  
    - Stable trends support reliable recommendation modeling  
    - Clean data improves accuracy of similarity and factorization methods  
    - Numeric features enable matrix-based recommendation techniques
    """)

else:
    st.info("👆 Upload a dataset to generate insights.")
