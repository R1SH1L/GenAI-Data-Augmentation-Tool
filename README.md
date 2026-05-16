# GenAI Data Augmentation Tool

Generate synthetic tabular data with CTGAN and evaluate a Random Forest model before and after augmentation.

## Requirements

- Python 3.9+ recommended
- See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (Streamlit)

```bash
streamlit run app.py
```

## Run (CLI)

1. Open `main.py` and set `DATA_PATH` to your CSV file.
2. Run:

```bash
python main.py
```

The script will:
- Load and preprocess the dataset
- Auto-detect a target column (low-cardinality heuristic)
- Train/evaluate Random Forest before augmentation
- Generate synthetic data with CTGAN
- Train/evaluate Random Forest after augmentation
- Save the augmented dataset to `output/augmented_data.csv`

Main pipeline logic lives in `src/pipeline.py`; `main.py` is a thin entry point that calls it.

## Run (Streamlit UI)

```bash
streamlit run app.py
```

In the UI you can:
- Upload a CSV
- Review a detected target column or choose a different one
- Generate synthetic data
- See Random Forest accuracy before and after augmentation
- Download the augmented dataset

## Data Samples

Sample CSVs are available in the `data/` folder.

## Output

Augmented data is saved to:

```
output/augmented_data.csv
```

## Example Output (CLI)

Shortened sample run:

```
✅ Dataset loaded successfully.
✅ Dataset preprocessed (missing values removed).
🔍 Analyzing columns to detect target...
🎯 Selected: 'label' (unique values: 2) - preferring last column position

🚀 MODEL TRAINING BEFORE AUGMENTATION
📊 Model Accuracy: 1.0000

⏳ Training CTGAN model (Generative AI)...
✅ Synthetic data generated using CTGAN (Generative AI).

🚀 MODEL TRAINING AFTER AUGMENTATION
📊 Model Accuracy: 0.7121
```

Notes:
- Accuracy can drop after augmentation depending on dataset size, label balance, and CTGAN quality. Treat the RF scores as a quick sanity check, not a benchmark.
- Target auto-detection uses a low-cardinality heuristic; override it in the UI if needed.
- SDV may emit a deprecation warning about `SingleTableMetadata`. It is safe to ignore for now, but you can upgrade the SDV API later if desired.
