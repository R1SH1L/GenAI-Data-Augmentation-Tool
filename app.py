import os
import streamlit as st
import pandas as pd
from src.dataset_handler import preprocess_dataset
from src.genai_tabular import augment_tabular
from src.model_train import train_and_eval

def detect_target_column(df):
    candidates = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count >= 2 and unique_count <= 10:
            candidates.append((col, unique_count))

    if candidates:
        candidates.sort(key=lambda x: (x[1], -list(df.columns).index(x[0])))
        return candidates[0][0], candidates

    return df.columns[-1], candidates

st.set_page_config(page_title="GenAI Data Augmentation Tool", layout="wide")

st.title("✨ GenAI Powered Data Augmentation (CTGAN Only)")
st.write("Upload a CSV dataset to generate synthetic tabular data using Generative AI.")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Original Dataset Preview")
    st.write(df.head())

    df = preprocess_dataset(df)

    detected_target, candidates = detect_target_column(df)
    detected_index = list(df.columns).index(detected_target)

    if candidates:
        st.info(
            f"Auto-detected target: {detected_target} (candidates: {[c[0] for c in candidates]})"
        )
    else:
        st.info(f"No low-cardinality target found. Using last column: {detected_target}")

    target_column = st.selectbox(
        "Select target column for RF evaluation",
        options=list(df.columns),
        index=detected_index
    )

    samples = st.slider("How many synthetic rows to generate?", 50, 500, 200)

    if st.button("🚀 Generate Synthetic Data"):
        st.subheader("📊 Random Forest Evaluation")
        acc_before = train_and_eval(df, target_column=target_column)

        df_aug = augment_tabular(df, samples)
        acc_after = train_and_eval(df_aug, target_column=target_column)

        st.success("✅ Data augmentation complete!")
        st.metric("RF Accuracy (before)", f"{acc_before:.4f}")
        st.metric("RF Accuracy (after)", f"{acc_after:.4f}")
        st.subheader("🔍 Augmented Dataset Preview")
        st.write(df_aug.tail())

        os.makedirs("output", exist_ok=True)
        df_aug.to_csv("output/augmented_data.csv", index=False)

        st.download_button(
            label="⬇ Download Augmented Dataset",
            data=df_aug.to_csv(index=False),
            file_name="augmented_data.csv",
            mime="text/csv"
        )
