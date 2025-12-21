import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Recommendation System", layout="wide")

# ---------------- LOAD BOOK DATASET ----------------
@st.cache_data
def load_books():
    return pd.read_csv("FBDAP Dataset.csv")

books_df = load_books()

# Clean trailing spaces only
books_df = books_df.applymap(
    lambda x: x.rstrip() if isinstance(x, str) else x
)

# ---------------- MAIN HEADING ----------------
st.title("📘 Recommendation System")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📚 Book Recommendation System",
    "📊 Dataset Summary",
    "📂 Custom Dataset Recommender"
])

# ==================================================
# TAB 1: BOOK RECOMMENDATION SYSTEM
# ==================================================
with tab1:
    st.subheader("Book Recommendation System")

    searchable_cols = books_df.select_dtypes(include="object").columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        search_col = st.selectbox(
            "Select search column",
            searchable_cols
        )

        # Example from dataset
        example_value = books_df[search_col].dropna().astype(str).iloc[0]
        user_input = st.text_input(
            f"Enter {search_col} (partial text allowed)",
            placeholder=f"e.g. {example_value}"
        )

    with col2:
        n_recs = st.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=50,
            value=5
        )

        rating_cols = books_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        rating_col = st.selectbox(
            "Select rating column (optional)",
            ["None"] + rating_cols
        )

    if user_input:
        mask = books_df[search_col].str.contains(
            user_input, case=False, na=False
        )
        results = books_df[mask]

        if rating_col != "None":
            results = results.sort_values(by=rating_col, ascending=False)

        results = results.head(n_recs)

        if results.empty:
            st.error("❌ No matching books found.")
        else:
            st.success(f"✅ Showing {len(results)} recommendations")
            st.dataframe(results.reset_index(drop=True))

    else:
        st.info("👆 Enter text to get book recommendations")

# ==================================================
# TAB 2: DATASET SUMMARY (NO MISSING VALUES SHOWN)
# ==================================================
with tab2:
    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", books_df.shape[0])
    c2.metric("Columns", books_df.shape[1])
    c3.metric("Categorical Columns", books_df.select_dtypes(include="object").shape[1])

    st.divider()

    st.markdown("### Column Information")
    col_info = pd.DataFrame({
        "Column": books_df.columns,
        "Data Type": books_df.dtypes.astype(str)
    })
    st.dataframe(col_info, use_container_width=True)

    st.divider()

    num_df = books_df.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        st.markdown("### Descriptive Statistics")
        st.dataframe(num_df.describe(), use_container_width=True)

    st.divider()

    st.markdown("### Top Values (Categorical)")
    cat_cols = books_df.select_dtypes(include="object").columns[:3]
    for col in cat_cols:
        st.markdown(f"**{col}**")
        st.dataframe(
            books_df[col].value_counts().head(5).reset_index(),
            use_container_width=True
        )

# ==================================================
# TAB 3: CUSTOM DATASET RECOMMENDER
# ==================================================
with tab3:
    st.subheader("Custom Dataset Recommender")

    uploaded = st.file_uploader(
        "Upload your dataset (CSV)",
        type="csv"
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        df = df.applymap(
            lambda x: x.rstrip() if isinstance(x, str) else x
        )

        st.success("Dataset uploaded successfully")

        text_cols = df.select_dtypes(include="object").columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            search_col = st.selectbox("Select column", text_cols)
            example = df[search_col].dropna().astype(str).iloc[0]
            text = st.text_input(
                "Enter search text",
                placeholder=f"e.g. {example}"
            )

        with col2:
            n = st.number_input(
                "Number of recommendations",
                min_value=1,
                max_value=50,
                value=5
            )

        if text:
            res = df[df[search_col].str.contains(text, case=False, na=False)]
            res = res.head(n)

            if res.empty:
                st.error("No matching records found.")
            else:
                st.dataframe(res.reset_index(drop=True))

        st.divider()
        st.markdown("### Custom Dataset Summary")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.dataframe(df.head(), use_container_width=True)
