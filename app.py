import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Book Recommendation System", layout="wide")

# ---------------- TITLE ----------------
st.title("📚 Book Recommendation System")

st.write(
    "This application recommends books **from the uploaded dataset** based on user-selected criteria "
    "such as author, title, or other attributes."
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Book Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # ---------------- TABS ----------------
    tab1, tab2 = st.tabs(["🔍 Recommendation System", "📊 Dataset Insights & Concepts"])

    # =========================================================
    # TAB 1: RECOMMENDATION SYSTEM
    # =========================================================
    with tab1:
        st.subheader("🔎 Get Book Recommendations")

        # Select column for recommendation
        selected_column = st.selectbox(
            "Select column to search (Author / Book / etc.)",
            df.columns
        )

        # User input (partial match allowed)
        user_input = st.text_input(
            f"Enter {selected_column} (partial text allowed)",
            placeholder="e.g. jk rowling"
        )

        # Number of recommendations
        top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5
        )

        # Optional rating filter
        rating_column = None
        for col in df.columns:
            if "rating" in col.lower():
                rating_column = col
                break

        if rating_column:
            min_rating = st.slider(
                f"Minimum {rating_column}",
                float(df[rating_column].min()),
                float(df[rating_column].max()),
                float(df[rating_column].min())
            )
        else:
            min_rating = None

        # ---------------- FILTER LOGIC ----------------
        if user_input:
            filtered_df = df[
                df[selected_column]
                .astype(str)
                .str.contains(user_input, case=False, na=False)
            ]

            if rating_column and min_rating is not None:
                filtered_df = filtered_df[filtered_df[rating_column] >= min_rating]

            if filtered_df.empty:
                st.warning("No matching books found in the dataset.")
            else:
                st.success(f"Showing top {top_n} recommendations")
                st.dataframe(filtered_df.head(top_n))

    # =========================================================
    # TAB 2: INSIGHTS & CONCEPTS
    # =========================================================
    with tab2:
        st.subheader("📊 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.divider()

        st.subheader("📌 Recommendation System Concepts (Gist)")
        st.markdown("""
        **Matrix Factorization**  
        Breaks user–item interaction matrices into latent factors to discover hidden preferences.

        **Content-Based Filtering**  
        Recommends books similar to those a user likes based on attributes such as author or genre.

        **Collaborative Filtering**  
        Uses similarities between users or items to generate recommendations.

        **Cosine Similarity**  
        Measures similarity between vectors and is widely used in recommendation scoring.

        **Text Embedding**  
        Converts textual data (titles, authors) into numerical vectors for similarity comparison.
        """)

        st.divider()

        st.subheader("📈 Basic Statistical Insights")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
        else:
            st.info("No numerical columns available for statistical analysis.")

else:
    st.info("👆 Please upload the dataset to start using the Book Recommendation System.")
