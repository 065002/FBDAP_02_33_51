import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="centered")

# ===================== TITLE =====================
st.title("📚 Simple Book Recommendation System")

st.write(
    "This application recommends books based on similarity using "
    "content-based filtering and cosine similarity."
)

st.divider()

# ===================== FILE UPLOAD =====================
uploaded_file = st.file_uploader("Upload Book Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.divider()

    # ===================== COLUMN SELECTION =====================
    st.subheader("Select Book Column")

    text_column = st.selectbox(
        "Choose the column containing book titles or descriptions",
        df.columns
    )

    # Clean data
    df = df[[text_column]].dropna()
    df[text_column] = df[text_column].astype(str)

    st.success("Column selected successfully")

    st.divider()

    # ===================== USER INPUT =====================
    st.subheader("Enter a Book Name")

    user_input = st.text_input("Type a book name from the dataset")

    if user_input:

        # ===================== VECTORIZE =====================
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df[text_column])

        # ===================== SIMILARITY =====================
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Find matching book
        if user_input not in df[text_column].values:
            st.error("Book not found in dataset. Please check spelling.")
        else:
            index = df[df[text_column] == user_input].index[0]
            scores = list(enumerate(similarity[index]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            st.subheader("📖 Recommended Books")

            recommendations = []
            for i in scores[1:6]:
                recommendations.append(df.iloc[i[0]][text_column])

            for book in recommendations:
                st.write("•", book)

else:
    st.info("Upload a CSV file to start recommending books.")
