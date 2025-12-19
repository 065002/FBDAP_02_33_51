import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config (looks professional)
st.set_page_config(page_title="Recommendation System App", layout="wide")

# Title
st.title("📊 Recommendation System – Data Insights Dashboard")

st.write(
    "This application allows users to upload a dataset and automatically generates "
    "useful insights, summaries, and visualizations."
)

st.divider()

# File upload
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # SECTION 1: Dataset Overview
    st.subheader("📌 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    st.divider()

    # SECTION 2: Preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    st.divider()

    # SECTION 3: Column Information
    st.subheader("🧾 Column Information")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values
    })
    st.table(col_info)

    st.divider()

    # SECTION 4: Missing Values
    st.subheader("⚠️ Missing Values Summary")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0]

    if len(missing_df) == 0:
        st.success("No missing values found in the dataset.")
    else:
        st.warning("Missing values detected:")
        st.dataframe(missing_df)

    st.divider()

    # SECTION 5: Statistical Summary
    st.subheader("📈 Statistical Summary (Numerical Columns)")
    st.dataframe(df.describe())

    st.divider()

    # SECTION 6: Visual Insights
    st.subheader("📊 Visual Insights")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_columns) > 0:
        selected_column = st.selectbox(
            "Select a numerical column to visualize",
            numeric_columns
        )

        fig, ax = plt.subplots()
        ax.hist(df[selected_column], bins=20)
        ax.set_title(f"Distribution of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
    else:
        st.info("No numerical columns available for visualization.")

else:
    st.info("👆 Please upload a CSV file to view insights.")
