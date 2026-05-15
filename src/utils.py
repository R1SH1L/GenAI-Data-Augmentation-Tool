from sklearn.preprocessing import LabelEncoder

def encode(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def detect_target_column(df, min_unique=2, max_unique=10):
    candidates = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count >= min_unique and unique_count <= max_unique:
            candidates.append((col, unique_count))

    if candidates:
        candidates.sort(key=lambda x: (x[1], -list(df.columns).index(x[0])))
        return candidates[0][0], candidates

    return df.columns[-1], candidates
