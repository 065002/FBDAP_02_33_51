import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommendation System Dashboard", layout="wide")

# =====================================================
# TITLE
# =====================================================
st.title("📚 Book Recommendation System – Analytics Dashboard")

st.write("""
This dashboard demonstrates a **Content-Based Recommendation System** using 
**Cosine Similarity and Text Embedding (TF-IDF)** along with analytical insights 
to support managerial decision-making.
""")

st.divider()

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("FBDAP Dataset.csv")

# Cleaning
df = df.dropna(subset=["title", "authors", "average_rating", "ratings_count"])
df["content"] = df["title"] + " " + df["authors"]

# =====================================================
# KPI SECTION
# =====================================================
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Books", df.shape[0])
col2.metric("Avg Rating", round(df["average_rating"].mean(), 2))
col3.metric("Total Ratings", int(df["ratings_count"].sum()))
col4.metric("Unique Authors", df["authors"].nunique())

st.divider()

# =====================================================
# DATA PREVIEW
# =====================================================
st.subheader("🔍 Dataset Preview")
st.dataframe(df[["title", "authors", "average_rating", "ratings_count"]].head(10))

st.divider()

# =====================================================
# 3D VISUALIZATION (MANAGER FRIENDLY)
# =====================================================
st.subheader("📈 3D Book Insights")

fig_3d = px.scatter_3d(
    df.sample(500),
    x="average_rating",
    y="ratings_count",
    z="  num_pages",
    color="average_rating",
    size="ratings_count",
    hover_name="title",
    title="3D View: Rating vs Popularity vs Book Size"
)

st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# =====================================================
# DISTRIBUTION INSIGHTS
# =====================================================
st.subheader("📊 Rating Distribution")

fig_hist = px.histogram(
    df,
    x="average_rating",
    nbins=20,
    title="Distribution of Book Ratings"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# =====================================================
# RECOMMENDATION SYSTEM (CORE PART)
# =====================================================
st.subheader("🤖 Book Recommendation System")

st.write("""
**Method Used:**  
- Text Embedding (TF-IDF)  
- Cosine Similarity  
- Content-Based Filtering  
""")

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

book_titles = df["title"].values
selected_book = st.selectbox("Select a book", book_titles)

if selected_book:
    idx = df[df["title"] == selected_book].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]

    st.write("### 📚 Recommended Books")
    st.dataframe(
        df.iloc[book_indices][
            ["title", "authors", "average_rating", "ratings_count"]
        ]
    )

st.divider()

# =====================================================
# MANAGERIAL INSIGHTS
# =====================================================
st.subheader("🧠 Managerial Insights")

st.markdown("""
- Books with **high ratings and high ratings count** indicate strong market acceptance.
- Recommendation system helps **personalize user experience**, increasing engagement.
- Popular authors and themes can guide **inventory and marketing strategies**.
- 3D visualization helps managers identify **high-value books quickly**.
""")

st.success("Dashboard ready for academic evaluation and managerial decision support.")
