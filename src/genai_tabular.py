from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

def augment_tabular(df, samples=200):
    print("\n⏳ Training CTGAN model (Generative AI)...")
    
    # Identify and exclude ID columns (based on data patterns only, not names)
    id_columns = []
    for col in df.columns:
        is_id = False
        
        # Only check integer columns for ID patterns (exclude floats like price, amount, etc.)
        if df[col].dtype in ['int64', 'int32']:
            
            # Check 1: All values are unique (every row has different value)
            if df[col].nunique() == len(df):
                is_id = True
            
            # Check 2: Values are sequential integers (1, 2, 3... or 0, 1, 2...)
            if not is_id:
                sorted_values = sorted(df[col].unique())
                # Check if it's a sequential series like [1,2,3...] or [0,1,2...]
                if len(sorted_values) == len(df) and sorted_values == list(range(sorted_values[0], sorted_values[0] + len(sorted_values))):
                    is_id = True
        
        if is_id:
            id_columns.append(col)
    
    if id_columns:
        print(f"🔍 Excluding ID columns from training: {id_columns}")
        df_train = df.drop(columns=id_columns)
    else:
        df_train = df.copy()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df_train)

    synthetic_data = synthesizer.sample(samples)
    
    # Add sequential IDs to synthetic data if ID columns were excluded
    if id_columns:
        for col in id_columns:
            # Generate sequential IDs starting after the original data
            start_id = int(df[col].max()) + 1
            synthetic_data[col] = range(start_id, start_id + len(synthetic_data))
    
    print("\n✅ Synthetic data generated using CTGAN (Generative AI).")

    return pd.concat([df, synthetic_data], ignore_index=True)
