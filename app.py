import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

st.title("📚 Book Recommendation System")

st.write(
    "This application recommends books **from the given dataset only**, "
    "based on user-selected preferences such as author, book title, or ratings."
)

st.divider()

# --------------------------------------------------
# LOAD DATA (PRE-UPLOADED DATASET)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# --------------------------------------------------
# SECTION 1: RECOMMENDATION SYSTEM
# --------------------------------------------------
st.header("🔍 Get Book Recommendations")

# Let user choose column
column_choice = st.selectbox(
    "Select the field you want to search by",
    df.columns
)

# User input (partial match allowed)
user_input = st.text_input(
    f"Enter {column_choice} (partial text allowed)"
)

# Number of recommendations
top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=5
)

# Optional rating filter
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

min_rating = None
if len(numeric_cols) > 0:
    rating_col = st.selectbox(
        "Select rating column (optional)",
        ["None"] + list(numeric_cols)
    )

    if rating_col != "None":
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
        df[column_choice]
        .astype(str)
        .str.contains(user_input, case=False, na=False)
    ]

    if min_rating is not None:
        filtered_df = filtered_df[filtered_df[rating_col] >= min_rating]

    if filtered_df.empty:
        st.warning("No matching books found.")
    else:
        st.success("Recommended books from the dataset:")
        st.dataframe(filtered_df.head(top_n))

st.divider()

# --------------------------------------------------
# SECTION 2: DATASET INSIGHTS
# --------------------------------------------------
st.header("📊 Dataset Insights")

# Basic overview
c1, c2, c3 = st.columns(3)
c1.metric("Total Books", df.shape[0])
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("📌 Topics Covered (Conceptual Overview)")
st.markdown("""
- **Matrix Factorization** – Latent factor discovery for recommendations  
- **Content-Based Filtering** – Recommendations based on item attributes  
- **Collaborative Filtering** – Similar user/item behavior  
- **Cosine Similarity** – Measures similarity between items/users  
- **Text Embedding** – Vectorizing text features for similarity matching  
""")

st.subheader("📈 Descriptive Statistics (Numerical Columns)")
numeric_df = df.select_dtypes(include=["int64", "float64"])
if not numeric_df.empty:
    st.dataframe(numeric_df.describe())
else:
    st.info("No numerical columns available.")

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10))
