import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# Normalize column names
df.columns = df.columns.str.strip()

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Dataset Summary", "📖 Recommendation System"])

# ==================================================
# TAB 1: DATASET SUMMARY (ONLY DATA INSIGHTS)
# ==================================================
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })
    st.dataframe(col_info, use_container_width=True)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        st.subheader("Descriptive Statistics")
        st.dataframe(numeric_cols.describe(), use_container_width=True)

# ==================================================
# TAB 2: BOOK RECOMMENDATION SYSTEM
# ==================================================
with tab2:
    st.subheader("Find Book Recommendations")

    # ---- USER INPUTS ----
    search_column = st.selectbox(
        "Select search type",
        options=df.columns.tolist()
    )

    search_text = st.text_input(
        f"Enter {search_column} (partial text allowed)",
        placeholder="e.g. J.K. Rowling"
    )

    num_recommendations = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=50,
        value=5
    )

    rating_column = st.selectbox(
        "Select rating column (optional)",
        options=["None"] + df.select_dtypes(include="number").columns.tolist()
    )

    # ---- PROCESSING ----
    if search_text.strip() != "":
        # Safe string matching
        filtered_df = df[
            df[search_column]
            .astype(str)
            .str.lower()
            .str.contains(search_text.lower(), na=False)
        ]

        if filtered_df.empty:
            st.error("❌ No matching books found.")
        else:
            # Optional sorting by rating
            if rating_column != "None":
                filtered_df = filtered_df.sort_values(
                    by=rating_column,
                    ascending=False
                )

            st.success(f"✅ {len(filtered_df)} matching books found")

            st.subheader("Recommended Books")
            st.dataframe(
                filtered_df.head(num_recommendations),
                use_container_width=True
            )

    else:
        st.info("👆 Enter text to get recommendations")
