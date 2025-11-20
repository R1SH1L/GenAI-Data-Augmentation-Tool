from sklearn.preprocessing import LabelEncoder

def encode(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df
