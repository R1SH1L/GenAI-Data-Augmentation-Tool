# GenAI Data Augmentation Tool

Generate synthetic tabular data with CTGAN and evaluate a Random Forest model before and after augmentation.

## Requirements

- Python 3.9+ recommended
- See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
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
