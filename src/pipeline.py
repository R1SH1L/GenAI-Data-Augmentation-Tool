import os
from src.dataset_handler import load_dataset, preprocess_dataset
from src.genai_tabular import augment_tabular
from src.model_train import train_and_eval
from src.utils import detect_target_column

def run_pipeline(data_path, samples=300, output_path="output/augmented_data.csv"):
    df = load_dataset(data_path)
    df = preprocess_dataset(df)

    print("Analyzing columns to detect target...")
    target_column, candidates = detect_target_column(df)

    if candidates:
        if len(candidates) > 1:
            print(f"Warning: Multiple low-cardinality columns found: {[c[0] for c in candidates]}")
            print(
                f"Selected: '{target_column}' (unique values: {candidates[0][1]}) - preferring last column position"
            )
        else:
            print(
                f"Auto-detected target column: '{target_column}' (unique values: {candidates[0][1]})"
            )
    else:
        print(f"Warning: No low-cardinality column found. Using last column: '{target_column}'")

    print("\nMODEL TRAINING BEFORE AUGMENTATION")
    acc_before = train_and_eval(df, target_column=target_column)

    df_aug = augment_tabular(df, samples=samples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_aug.to_csv(output_path, index=False)
    print(f"\nAugmented data saved to: {output_path}")

    print("\nMODEL TRAINING AFTER AUGMENTATION")
    acc_after = train_and_eval(df_aug, target_column=target_column)

    print(f"\nRF Accuracy (before): {acc_before:.4f}")
    print(f"RF Accuracy (after):  {acc_after:.4f}")

    return {
        "target_column": target_column,
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "output_path": output_path,
    }
