import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    print("\n✅ Dataset loaded successfully.")
    print(df.head())
    return df

def preprocess_dataset(df):
    df = df.dropna().reset_index(drop=True)
    print("\n✅ Dataset preprocessed (missing values removed).")
    return df
