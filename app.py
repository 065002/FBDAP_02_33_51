import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Book Recommendation System", layout="centered")

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")

st.write(
    "This application recommends books **directly from the dataset** "
    "based on user input such as author name, book title, or any selected column."
)

st.divider()

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# ---------------- RECOMMENDATION SYSTEM ----------------
st.subheader("🔎 Get Book Recommendations")

st.write("Select a column and enter a value to get matching book recommendations.")

# Column selection
selected_column = st.selectbox(
    "Select the field you want to search",
    df.columns
)

# User input
user_input = st.text_input(
    f"Enter value for {selected_column} (partial match allowed)"
)

# Recommendation logic
if user_input:
    recommendations = df[
        df[selected_column]
        .astype(str)
        .str.contains(user_input, case=False, na=False)
    ]

    if len(recommendations) > 0:
        st.success(f"Found {len(recommendations)} matching recommendations")
        st.dataframe(recommendations)
    else:
        st.warning("No matching records found.")

st.divider()

# ---------------- INSIGHTS SECTION ----------------
st.subheader("📊 Dataset Insights")

# Basic metrics
col1, col2 = st.columns(2)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Columns", df.shape[1])

st.divider()

# Preview
st.write("🔍 Dataset Preview")
st.dataframe(df.head())

st.divider()

# Concepts overview
st.subheader("📌 Concepts Covered (Gist)")

st.markdown("""
- **Content-Based Filtering**: Recommends books based on matching attributes such as author or title.  
- **Collaborative Filtering (Conceptual)**: Uses similarity across users or items (explained conceptually).  
- **Cosine Similarity**: Measures similarity between vectors in recommendation systems.  
- **Matrix Factorization**: Reduces user–item matrices into latent factors for recommendations.  
- **Text Embedding**: Converts book titles or descriptions into numerical form for similarity matching.
""")

st.info("This deployment demonstrates a **data-driven recommendation system** suitable for managerial decision support.")
