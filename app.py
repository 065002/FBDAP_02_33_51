import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")

# ===================== LOAD MAIN DATASET =====================
@st.cache_data
def load_main_data():
    return pd.read_csv("FBDAP Dataset.csv")

df_main = load_main_data()

# ===================== HELPER FUNCTIONS =====================
def clean_text(series):
    return series.astype(str).str.lower().str.strip()

def get_recommendations(df, col, query, top_n, rating_col=None):
    df = df.copy()
    df[col] = clean_text(df[col])

    query = query.lower().strip()

    matches = df[df[col].str.contains(query, na=False)]

    if rating_col and rating_col in df.columns:
        matches = matches.sort_values(by=rating_col, ascending=False)

    return matches.head(top_n)

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs([
    "📚 Book Recommendation System",
    "📊 Dataset Summary",
    "📂 Custom Dataset Recommender"
])

# ============================================================
# TAB 1: BOOK RECOMMENDATION SYSTEM (MAIN)
# ============================================================
with tab1:
    st.header("📚 Book Recommendation System")

    text_columns = df_main.select_dtypes(include="object").columns.tolist()
    numeric_columns = df_main.select_dtypes(include=["int64", "float64"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        search_col = st.selectbox("Select search column", text_columns)

        example_value = df_main[search_col].dropna().iloc[0]
        user_input = st.text_input(
            f"Enter {search_col} (partial text allowed)",
            placeholder=f"e.g. {example_value}"
        )

    with col2:
        top_n = st.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5
        )

        rating_col = st.selectbox(
            "Select rating column (optional)",
            ["None"] + numeric_columns
        )

    if user_input:
        results = get_recommendations(
            df_main,
            search_col,
            user_input,
            top_n,
            rating_col if rating_col != "None" else None
        )

        if len(results) == 0:
            st.error("❌ No matching books found.")
        else:
            st.success(f"✅ Showing {len(results)} recommendations")
            st.dataframe(results.reset_index(drop=True))

    else:
        st.info("👆 Enter text to get book recommendations")

# ============================================================
# TAB 2: DATASET SUMMARY (ONLY SUMMARY — NO THEORY)
# ============================================================
with tab2:
    st.header("📊 Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df_main.shape[0])
    c2.metric("Total Columns", df_main.shape[1])
    c3.metric("Missing Values", int(df_main.isnull().sum().sum()))

    st.subheader("🔍 Preview")
    st.dataframe(df_main.head())

    st.subheader("📈 Numerical Summary")
    num_df = df_main.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        st.dataframe(num_df.describe())
    else:
        st.info("No numerical columns available.")

# ============================================================
# TAB 3: CUSTOM DATASET RECOMMENDER
# ============================================================
with tab3:
    st.header("📂 Custom Dataset Recommendation System")

    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

    if uploaded_file:
        df_custom = pd.read_csv(uploaded_file)

        text_cols = df_custom.select_dtypes(include="object").columns.tolist()
        num_cols = df_custom.select_dtypes(include=["int64", "float64"]).columns.tolist()

        st.subheader("📊 Custom Dataset Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_custom.shape[0])
        c2.metric("Columns", df_custom.shape[1])
        c3.metric("Missing Values", int(df_custom.isnull().sum().sum()))

        st.dataframe(df_custom.head())

        st.subheader("🔎 Get Recommendations")

        search_col = st.selectbox("Search column", text_cols)
        example = df_custom[search_col].dropna().iloc[0]

        query = st.text_input(
            "Enter value (partial allowed)",
            placeholder=f"e.g. {example}"
        )

        top_n = st.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5,
            key="custom_top"
        )

        rating_col = st.selectbox(
            "Rating column (optional)",
            ["None"] + num_cols,
            key="custom_rating"
        )

        if query:
            results = get_recommendations(
                df_custom,
                search_col,
                query,
                top_n,
                rating_col if rating_col != "None" else None
            )

            if len(results) == 0:
                st.error("❌ No matching records found.")
            else:
                st.success("✅ Recommendations")
                st.dataframe(results.reset_index(drop=True))
