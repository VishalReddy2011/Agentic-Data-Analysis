import pandas as pd
import numpy as np
from typing import Dict, List
from pydantic import BaseModel

class DescriptiveStats(BaseModel):
    file_length: int
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    numeric_stats_df: pd.DataFrame
    categorical_stats_df: pd.DataFrame
    value_counts: Dict[str, Dict[str, int]]
    model_config = {"arbitrary_types_allowed": True}

def generate_descriptive_stats(csv_path: str) -> DescriptiveStats:
    df = pd.read_csv(csv_path)

    file_length = len(df)
    total_columns = len(df.columns)

    numeric_df = df.select_dtypes(include=[np.number]).describe()
    categorical_df = df.select_dtypes(include=["object", "category"]).describe()

    value_counts_map = {}
    for col in categorical_df.columns:
        s = df[col].value_counts().head(20)
        value_counts_map[col] = {str(k): int(v) for k, v in s.items()}

    return DescriptiveStats(
        file_length=file_length,
        total_columns=total_columns,
        numeric_columns=list(numeric_df.columns),
        categorical_columns=list(categorical_df.columns),
        numeric_stats_df=numeric_df,
        categorical_stats_df=categorical_df,
        value_counts=value_counts_map
    )
