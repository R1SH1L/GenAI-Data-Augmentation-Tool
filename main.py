from src.pipeline import run_pipeline

DATA_PATH = "data/sample_student_performance.csv"   # put your file here

run_pipeline(DATA_PATH, samples=300, output_path="output/augmented_data.csv")
