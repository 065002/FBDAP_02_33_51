import streamlit as st
import pandas as pd

# App title
st.title("Recommendation System Project")

# Project description
st.write(
    "This application demonstrates a deployed data-driven project using Streamlit Cloud. "
    "Users can upload their own dataset and view it directly in the browser."
)

# File uploader
st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df)
    
    st.success("Dataset uploaded successfully!")
else:
    st.info("Please upload a CSV file to proceed.")
