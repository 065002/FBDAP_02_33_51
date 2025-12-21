import streamlit as st
import pandas as pd
from difflib import get_close_matches

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")

st.write(
    "This application allows users to explore dataset insights and "
    "get book recommendations using partial matching and similarity logic."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# ---------------- SIDEBAR NAVIGATION ----------------
section = st.sidebar.radio(
    "Go to",
    ["Insights", "Book Recommendation System"]
)

# =====================================================
# ======================= INSIGHTS ====================
# =====================================================
if section == "Insights":

    st.header("📊 Dataset Insights & Concepts")

    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("🔍 Preview")
    st.dataframe(df.head())

    st.divider()

    st.subheader("📌 Recommendation System Concepts (Gist)")

    st.markdown("""
**Matrix Factorization**  
Breaks user–item interaction data into latent factors to uncover hidden preferences.

**Content-Based Filtering**  
Recommends items similar to what the user already likes based on attributes.

**Collaborative Filtering**  
Uses behavior of similar users to generate recommendations.

**Cosine Similarity**  
Measures similarity between vectors to find related users or items.

**Text Embedding**  
Converts textual information (book titles, authors) into numerical form for comparison.
""")

# =====================================================
# ============ BOOK RECOMMENDATION SYSTEM ==============
# =====================================================
if section == "Book Recommendation System":

    st.header("📖 Get Book Recommendations")

    st.write(
        "Select a column, enter a keyword (partial name allowed), "
        "and get similar book recommendations."
    )

    # ---------------- COLUMN SELECTION ----------------
    text_columns = df.select_dtypes(include="object").columns.tolist()

    if len(text_columns) == 0:
        st.error("No text columns available in this dataset.")
    else:
        selected_column = st.selectbox(
            "Select the column to search",
            text_columns
        )

        user_input = st.text_input(
            f"Enter {selected_column} (partial name allowed)"
        )

        if user_input:
            values = df[selected_column].dropna().astype(str).unique().tolist()

            matches = get_close_matches(
                user_input,
                values,
                n=10,
                cutoff=0.3   # allows partial match
            )

            if matches:
                st.subheader("✅ Recommended Results")
                result_df = df[df[selected_column].isin(matches)]
                st.dataframe(result_df)
            else:
                st.warning("No matching results found. Try another keyword.")
