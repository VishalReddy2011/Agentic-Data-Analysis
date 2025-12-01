from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import os
import contextlib
import io


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    unique_count: Optional[int] = None
    sample_values: Optional[List[str]] = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class ProfileOutput(BaseModel):
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    head: List[Dict[str, Any]]

    model_config = {"extra": "forbid", "validate_assignment": True}


# -------------------------------------------------------------------
# SIMPLE, DIRECT, UNSANDBOXED EXECUTION
# -------------------------------------------------------------------

def get_csv_profile(csv_path: str) -> Dict:

    df = pd.read_csv(csv_path)
    row_count = int(len(df))
    column_count = int(len(df.columns))

    columns: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            dtype = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(ser):
            dtype = "datetime"
        else:
            dtype = "categorical"

        unique_count = int(ser.nunique(dropna=True))
        sample = ser.dropna().astype(str).head(5).tolist()
        columns[col] = ColumnProfile(
            name=col,
            dtype=dtype,
            unique_count=unique_count,
            sample_values=sample
        )

    head_records = df.head(5).to_dict(orient="records")

    return {
        "row_count":row_count,
        "column_count":column_count,
        "columns":columns,
        "head":head_records
    }


def run_python_code(payload: Dict) -> Dict:
    """Executes LLM-generated Python code directly. No sandbox."""

    df = pd.read_csv(payload['csv_path'])
    local_vars = {"df": df}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(payload['code'], {}, local_vars)
    stdout = buf.getvalue()

    img_path = None
    if plt.get_fignums():
        fig = plt.gcf()
        filename = f"{payload["session_id"]}_plot_{payload["iteration_count"]}.png"
        img_path = os.path.join("./static/images", filename)
        fig.savefig(img_path, bbox_inches="tight")
    plt.close()

    return {
        "stdout" : stdout if stdout else None,
        "img_path" : img_path,
        "result" : local_vars.get("result")
    }
