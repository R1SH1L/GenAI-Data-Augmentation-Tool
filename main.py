import os
from src.dataset_handler import load_dataset, preprocess_dataset
from src.genai_tabular import augment_tabular
from src.model_train import train_and_eval

DATA_PATH = "data/sample_student_performance.csv"   # put your file here

df = load_dataset(DATA_PATH)
df = preprocess_dataset(df)

# Auto-detect target column by low cardinality analysis
print("🔍 Analyzing columns to detect target...")
TARGET_COLUMN = None
candidates = []

# Find all columns with low cardinality (2-10 unique values)
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count >= 2 and unique_count <= 10:
        candidates.append((col, unique_count))

if candidates:
    # Sort by cardinality (lowest first), then prefer last column if tie
    candidates.sort(key=lambda x: (x[1], -list(df.columns).index(x[0])))
    TARGET_COLUMN = candidates[0][0]
    
    if len(candidates) > 1:
        print(f"⚠️ Multiple low-cardinality columns found: {[c[0] for c in candidates]}")
        print(f"🎯 Selected: '{TARGET_COLUMN}' (unique values: {candidates[0][1]}) - preferring last column position")
    else:
        print(f"🎯 Auto-detected target column: '{TARGET_COLUMN}' (unique values: {candidates[0][1]})")
else:
    # Fallback to last column if no low-cardinality column found
    TARGET_COLUMN = df.columns[-1]
    print(f"⚠️ No low-cardinality column found. Using last column: '{TARGET_COLUMN}'")

print("\n🚀 MODEL TRAINING BEFORE AUGMENTATION")
acc_before = train_and_eval(df, target_column=TARGET_COLUMN)

df_aug = augment_tabular(df, samples=300)

# Save augmented data
os.makedirs("output", exist_ok=True)
output_path = "output/augmented_data.csv"
df_aug.to_csv(output_path, index=False)
print(f"\n💾 Augmented data saved to: {output_path}")

print("\n🚀 MODEL TRAINING AFTER AUGMENTATION")
acc_after = train_and_eval(df_aug, target_column=TARGET_COLUMN)

print(f"\n✅ RF Accuracy (before): {acc_before:.4f}")
print(f"✅ RF Accuracy (after):  {acc_after:.4f}")
