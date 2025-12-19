import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recommendation System Dashboard", layout="wide")

# ===================== TITLE =====================
st.title("📊 Recommendation System – Analytics & Insights Dashboard")

st.write(
    "This deployed application provides a high-level demonstration of recommendation system concepts "
    "along with automated statistical and analytical insights from user-uploaded datasets."
)

st.divider()

# ===================== TOPICS EXPLORED =====================
st.subheader("📌 Topics Explored (Conceptual Overview)")

st.markdown("""
**Matrix Factorization**  
Used to decompose large user–item interaction matrices into latent factors for uncovering hidden patterns.

**Content-Based Filtering**  
Recommends items similar to user preferences based on item attributes.

**Collaborative Filtering**  
Generates recommendations using user behavior patterns and similarities across users.

**Cosine Similarity**  
Measures similarity between vectors and is widely used for recommendation scoring.

**Text Embedding**  
Transforms textual data into numerical vectors for similarity comparison and recommendation tasks.
""")

st.divider()

# ===================== FILE UPLOAD =====================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ===================== DATASET OVERVIEW =====================
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])

    st.divider()

    # ===================== PREVIEW =====================
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    st.divider()

    # ===================== DESCRIPTIVE STATISTICS =====================
    st.subheader("📈 Descriptive Statistics")
    st.write(
        "Provides summary statistics such as mean, standard deviation, minimum and maximum values "
        "to understand the overall data distribution."
    )
    st.dataframe(df.describe())

    st.divider()

    # ===================== LINEAR ALGEBRA INSIGHTS =====================
    st.subheader("🧮 Linear Algebra Insights (Matrix Operations)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] > 0:
        st.write(
            "Numerical columns form a data matrix that can be used for matrix operations "
            "such as dot products and transformations in recommendation systems."
        )
        st.write("**Matrix Shape:**", numeric_df.shape)
    else:
        st.info("No numerical columns available for matrix operations.")

    st.divider()

    # ===================== TIME SERIES INSIGHTS =====================
    st.subheader("⏳ Time Series Insights (Moving Average)")

    if numeric_df.shape[1] > 0:
        ts_column = st.selectbox(
            "Select a numerical column for Moving Average analysis",
            numeric_df.columns
        )

        window = st.slider("Moving Average Window Size", 2, 10, 3)

        df["Moving_Avg"] = df[ts_column].rolling(window=window).mean()

        fig, ax = plt.subplots()
        ax.plot(df[ts_column], label="Original Data")
        ax.plot(df["Moving_Avg"], label="Moving Average")
        ax.set_title("Moving Average Trend Analysis")
        ax.legend()

        st.pyplot(fig)
    else:
        st.info("Time series analysis requires numerical data.")

    st.divider()

    # ===================== PROBABILITY & STATISTICS =====================
    st.subheader("📐 Probability & Statistical Insights")

    st.write("""
    - **Descriptive Statistics:** Central tendency and dispersion of data  
    - **Probability & Distribution:** Understanding spread and likelihood patterns  
    - **Inferential Statistics:** Basis for hypothesis testing and confidence estimation  
    - **Regression:** Relationship analysis between dependent and independent variables
    """)

    if numeric_df.shape[1] >= 2:
        x_col = numeric_df.columns[0]
        y_col = numeric_df.columns[1]

        coeff = np.polyfit(df[x_col], df[y_col], 1)
        st.write(f"**Regression Insight:** `{y_col} = {coeff[0]:.2f} × {x_col} + {coeff[1]:.2f}`")

    st.divider()

    # ===================== VISUAL INSIGHTS =====================
    st.subheader("📊 Visual Distribution Analysis")

    if numeric_df.shape[1] > 0:
        selected_col = st.selectbox("Select column for distribution", numeric_df.columns)

        fig, ax = plt.subplots()
        ax.hist(df[selected_col], bins=20)
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

else:
    st.info("👆 Upload a dataset to generate insights.")
