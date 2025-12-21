import streamlit as st
import pandas as pd
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("FBDAP Dataset.csv")

    # Trim spaces ONLY at the end (important requirement)
    df = df.applymap(
        lambda x: x.rstrip() if isinstance(x, str) else x
    )

    return df

df = load_data()

# -------------------- TABS (ORDER MATTERS) --------------------
tab_rec, tab_summary = st.tabs([
    "📚 Book Recommendation System",
    "📊 Dataset Summary"
])

# ==============================================================
# TAB 1: BOOK RECOMMENDATION SYSTEM
# ==============================================================
with tab_rec:

    st.header("📚 Book Recommendation System")

    # Select column
    search_col = st.selectbox(
        "Select search column",
        options=df.columns
    )

    # Example value from selected column
    example_value = (
        df[search_col]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    user_input = st.text_input(
        f"Enter {search_col} (partial match allowed)",
        placeholder=f"e.g. {example_value}"
    )

    top_n = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5
    )

    rating_col = st.selectbox(
        "Select rating column (optional)",
        options=["None"] + list(df.select_dtypes(include=np.number).columns)
    )

    st.divider()

    if user_input.strip() != "":
        # Case-insensitive partial matching
        mask = df[search_col].astype(str).str.contains(
            user_input,
            case=False,
            na=False
        )

        results = df[mask].copy()

        if rating_col != "None" and rating_col in results.columns:
            results = results.sort_values(
                by=rating_col,
                ascending=False
            )

        results = results.head(top_n)

        if len(results) == 0:
            st.error("❌ No matching records found in the dataset.")
        else:
            st.success(f"✅ Showing {len(results)} recommendation(s)")
            st.dataframe(results.reset_index(drop=True))

    else:
        st.info("👆 Enter text to get recommendations")

# ==============================================================
# TAB 2: DATASET SUMMARY (NO PROJECT TEXT HERE)
# ==============================================================
with tab_summary:

    st.header("📊 Dataset Summary")

    # Overview metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])

    # Count NULLs (spaces already trimmed)
    null_count = df.isna().sum().sum()
    c3.metric("Missing Values", int(null_count))

    st.divider()

    # Column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum().values
    })
    st.dataframe(col_info)

    st.divider()

    # Numeric summary
    num_cols = df.select_dtypes(include=np.number)

    if not num_cols.empty:
        st.subheader("Descriptive Statistics (Numeric Columns)")
        st.dataframe(num_cols.describe())

    st.divider()

    # Categorical summary
    cat_cols = df.select_dtypes(include="object")

    if not cat_cols.empty:
        st.subheader("Top Values (Categorical Columns)")
        for col in cat_cols.columns:
            st.write(f"**{col}**")
            st.dataframe(
                cat_cols[col]
                .value_counts()
                .head(5)
                .reset_index()
                .rename(columns={"index": col, col: "Count"})
            )
