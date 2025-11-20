import streamlit as st
import pandas as pd
from src.dataset_handler import preprocess_dataset
from src.genai_tabular import augment_tabular

st.set_page_config(page_title="GenAI Data Augmentation Tool", layout="wide")

st.title("✨ GenAI Powered Data Augmentation (CTGAN Only)")
st.write("Upload a CSV dataset to generate synthetic tabular data using Generative AI.")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Original Dataset Preview")
    st.write(df.head())

    df = preprocess_dataset(df)

    samples = st.slider("How many synthetic rows to generate?", 50, 500, 200)

    if st.button("🚀 Generate Synthetic Data"):
        df_aug = augment_tabular(df, samples)

        st.success("✅ Data augmentation complete!")
        st.subheader("🔍 Augmented Dataset Preview")
        st.write(df_aug.tail())

        df_aug.to_csv("output/augmented_data.csv", index=False)

        st.download_button(
            label="⬇ Download Augmented Dataset",
            data=df_aug.to_csv(index=False),
            file_name="augmented_data.csv",
            mime="text/csv"
        )
