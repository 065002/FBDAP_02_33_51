import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recommendation System Dashboard", layout="wide")

# ===================== TITLE =====================
st.title("📊 Recommendation System – Analytics & Insights Dashboard")

st.write(
    "This deployed application demonstrates recommendation system concepts "
    "and generates statistical insights and visualizations from user-uploaded datasets."
)

st.divider()

# ===================== TOPICS EXPLORED =====================
st.subheader("📌 Topics Explored (Conceptual Overview)")

st.markdown("""
**Matrix Factorization** – Decomposes user–item matrices into latent factors  
**Content-Based Filtering** – Recommends items based on item features  
**Collaborative Filtering** – Uses user similarity and behavior patterns  
**Cosine Similarity** – Measures similarity between vectors  
**Text Embedding** – Converts text into numerical representations
""")

st.divider()

# ===================== FILE UPLOAD =====================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    categorical_df = df.select_dtypes(include=["object"])

    # ===================== DATASET OVERVIEW =====================
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])

    st.divider()

    # ===================== DATA PREVIEW =====================
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    st.divider()

    # ===================== DESCRIPTIVE STATS =====================
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe())

    st.divider()

    # ===================== BOX PLOT =====================
    st.subheader("📦 Box Plot (Outliers & Spread)")

    if numeric_df.shape[1] > 0:
        box_col = st.selectbox("Select column for boxplot", numeric_df.columns)
        fig, ax = plt.subplots()
        ax.boxplot(df[box_col].dropna())
        ax.set_title(f"Box Plot of {box_col}")
        st.pyplot(fig)
    else:
        st.info("No numerical columns available for boxplot.")

    st.divider()

    # ===================== CORRELATION HEATMAP =====================
    st.subheader("🔥 Correlation Heatmap")

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Matrix", pad=20)
        st.pyplot(fig)
    else:
        st.info("Correlation requires at least two numerical variables.")

    st.divider()

    # ===================== SCATTER PLOT =====================
    st.subheader("📍 Scatter Plot (Relationship Analysis)")

    if numeric_df.shape[1] >= 2:
        x_axis = st.selectbox("X-axis", numeric_df.columns, key="x")
        y_axis = st.selectbox("Y-axis", numeric_df.columns, key="y")

        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.info("Scatter plot requires at least two numerical columns.")

    st.divider()

    # ===================== TIME SERIES =====================
    st.subheader("⏳ Time Series – Moving Average")

    if numeric_df.shape[1] > 0:
        ts_col = st.selectbox("Select column for trend analysis", numeric_df.columns)
        window = st.slider("Moving Average Window", 2, 10, 3)

        ma = df[ts_col].rolling(window=window).mean()

        fig, ax = plt.subplots()
        ax.plot(df[ts_col], label="Original")
        ax.plot(ma, label="Moving Average")
        ax.legend()
        ax.set_title("Trend Analysis")
        st.pyplot(fig)
    else:
        st.info("No numerical data for time series analysis.")

    st.divider()

    # ===================== CATEGORICAL DISTRIBUTION =====================
    st.subheader("🧩 Categorical Distribution")

    if categorical_df.shape[1] > 0:
        cat_col = st.selectbox("Select categorical column", categorical_df.columns)
        counts = df[cat_col].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Category Count: {cat_col}")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No categorical columns found.")

else:
    st.info("👆 Upload a dataset to generate insights and visualizations.")
