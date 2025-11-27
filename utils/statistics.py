import pandas as pd
import numpy as np

def generate_descriptive_stats(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    file_length = len(df)
    total_columns = len(df.columns)

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    return {
        "file_length": file_length,
        "total_columns": total_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }
