import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="centered")

# ===================== TITLE =====================
st.title("📚 Book Recommendation System")

st.write(
    "Enter a book name (full or partial). "
    "The system will recommend similar books using text similarity."
)

st.divider()

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("FBDAP Dataset.csv")

df = load_data()

# ===================== COLUMN SELECTION =====================
st.subheader("📄 Dataset Loaded")

st.write("Select the column that contains **Book Titles**:")

book_col = st.selectbox(
    "Book Title Column",
    df.columns
)

# ===================== CLEAN DATA =====================
books = df[[book_col]].dropna()
books[book_col] = books[book_col].astype(str)

# ===================== TEXT VECTORISATION =====================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books[book_col])

# ===================== USER INPUT =====================
st.divider()
user_input = st.text_input("🔍 Enter a book name")

# ===================== RECOMMENDATION LOGIC =====================
if user_input:
    input_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-6:][::-1]

    recommendations = books.iloc[top_indices][book_col].values

    st.subheader("📖 Recommended Books")
    for book in recommendations:
        st.write("•", book)
