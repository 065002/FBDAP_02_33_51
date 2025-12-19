import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Recommendation System Dashboard",
    layout="wide"
)

# ================= TITLE =================
st.title("📊 Recommendation System – Analytics & Decision Dashboard")

st.write(
    "This interactive dashboard helps managers and analysts quickly derive insights "
    "from data and understand recommendation system concepts."
)

st.divider()

# ================= CONCEPTUAL SECTION =================
st.subheader("📌 Recommendation System Concepts (Gist)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Matrix Factorization**  
    Decomposes user–item interaction matrices to uncover latent patterns.
    
    **Content-Based Filtering**  
    Recommends items based on similarity to user preferences.
    """)

with col2:
    st.markdown("""
    **Collaborative Filtering**  
    Uses behavior of similar users to generate recommendations.
    
    **Cosine Similarity**  
    Measures similarity between users/items using vector angles.
    
    **Text Embedding**  
    Converts text into numeric vectors for similarity & recommendations.
    """)

st.divider()

# ================= FILE UPLOAD =================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ================= DATA OVERVIEW =================
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.divider()

    # ================= DATA PREVIEW =================
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10))

    st.divider()

    # ================= DESCRIPTIVE STATS =================
    st.subheader("📈 Descriptive Statistics")
    st.write("Summary statistics help managers understand central tendency and spread.")
    st.dataframe(df.describe())

    st.divider()

    # ================= NUMERIC DATA =================
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # ================= DISTRIBUTION PLOT =================
    st.subheader("📊 Distribution Analysis")
    if not numeric_df.empty:
        col = st.selectbox("Select column", numeric_df.columns)
        fig, ax = plt.subplots()
        ax.hist(numeric_df[col], bins=20)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns found.")

    st.divider()

    # ================= TIME SERIES (MOVING AVERAGE) =================
    st.subheader("⏳ Trend Analysis (Moving Average)")
    if not numeric_df.empty:
        ts_col = st.selectbox("Select column for trend", numeric_df.columns)
        window = st.slider("Window size", 2, 10, 3)

        ma = numeric_df[ts_col].rolling(window).mean()

        fig, ax = plt.subplots()
        ax.plot(numeric_df[ts_col], label="Original")
        ax.plot(ma, label="Moving Avg")
        ax.legend()
        st.pyplot(fig)

    st.divider()

    # ================= REGRESSION INSIGHT =================
    st.subheader("📐 Regression Insight")
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0]
        y = numeric_df.iloc[:, 1]
        coef = np.polyfit(x, y, 1)
        st.write(f"Regression Equation: **y = {coef[0]:.2f}x + {coef[1]:.2f}**")
    else:
        st.info("Need at least 2 numeric columns.")

    st.divider()

    # ================= 3D VISUALIZATION =================
    st.subheader("🧊 3D Data Visualization")
    if numeric_df.shape[1] >= 3:
        x = numeric_df.iloc[:, 0]
        y = numeric_df.iloc[:, 1]
        z = numeric_df.iloc[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel(numeric_df.columns[0])
        ax.set_ylabel(numeric_df.columns[1])
        ax.set_zlabel(numeric_df.columns[2])
        st.pyplot(fig)
    else:
        st.info("Need at least 3 numeric columns for 3D plot.")

    st.divider()

    # ================= MANAGERIAL INSIGHTS =================
    st.subheader("🧠 Managerial Insights")
    st.markdown("""
    - Identifies key trends and variability in data  
    - Highlights relationships between variables  
    - Supports data-driven recommendation strategies  
    - Reduces manual analysis effort for decision-makers  
    """)

else:
    st.info("👆 Upload a dataset to generate insights.")
