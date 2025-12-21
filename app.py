import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered"
)

st.title("📚 Book Recommendation System")

st.write(
    "This application recommends books **only from the given dataset** "
    "based on user-selected criteria such as author, title, or publisher."
)

st.divider()

# ---------------- LOAD DATA (PRE-UPLOADED) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("FBDAP Dataset.csv", encoding="latin-1")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ---------------- USER INPUTS ----------------
st.subheader("🔎 Find Book Recommendations")

search_column = st.selectbox(
    "Select search type",
    options=df.columns,
    index=df.columns.get_loc("authors") if "authors" in df.columns else 0
)

search_text = st.text_input(
    f"Enter {search_column} (partial text allowed)"
)

num_recs = st.number_input(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=5
)

rating_column = st.selectbox(
    "Select rating column (optional)",
    options=["None"] + df.select_dtypes(include="number").columns.tolist()
)

st.divider()

# ---------------- DATA CLEANING ----------------
df[search_column] = df[search_column].astype(str).str.lower().str.strip()

# ---------------- RECOMMENDATION LOGIC ----------------
if search_text:
    search_text = search_text.lower().strip()

    results = df[df[search_column].str.contains(search_text, na=False)]

    if rating_column != "None":
        results = results.sort_values(by=rating_column, ascending=False)

    results = results.head(num_recs)

    if results.empty:
        st.error("❌ No matching books found.")
    else:
        st.success(f"✅ {len(results)} recommendation(s) found")

        display_cols = [
            col for col in ["title", "authors", "publisher", rating_column]
            if col in results.columns and col != "None"
        ]

        st.dataframe(results[display_cols].reset_index(drop=True))

else:
    st.info("👆 Enter a search value to get recommendations.")

# ---------------- INSIGHTS SECTION ----------------
st.divider()
st.subheader("📊 Dataset Insights")

st.markdown("""
**Topics Covered in This Project**
- Matrix Factorization (conceptual)
- Content-Based Filtering
- Collaborative Filtering
- Cosine Similarity
- Text Embedding (conceptual)

This application demonstrates **content-based filtering** using text matching.
""")

c1, c2, c3 = st.columns(3)
c1.metric("Total Books", len(df))
c2.metric("Unique Authors", df["authors"].nunique() if "authors" in df.columns else "NA")
c3.metric("Total Columns", df.shape[1])
