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


class ProfileInput(BaseModel):
    file_path: str

    model_config = {"extra": "forbid", "validate_assignment": True}


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


class CodeInput(BaseModel):
    code: str
    csv_path: Optional[str] = None
    session_id: Optional[str] = None
    iteration_count: Optional[int] = None
    work_dir: Optional[str] = "static/images"

    model_config = {"extra": "forbid", "validate_assignment": True}


class CodeOutput(BaseModel):
    status: str
    stdout: Optional[str] = None
    error: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    result: Optional[Any] = None

    model_config = {"extra": "forbid", "validate_assignment": True}


# -------------------------------------------------------------------
# SIMPLE, DIRECT, UNSANDBOXED EXECUTION
# -------------------------------------------------------------------

def get_csv_profile(input_data: Dict) -> Dict:
    """Reads CSV and returns metadata."""
    try:
        inp = ProfileInput.model_validate(input_data)
    except ValidationError as e:
        return {"error": f"ProfileInput validation error: {e}"}

    df = pd.read_csv(inp.file_path)
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

    out = ProfileOutput(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        head=head_records
    )
    return out.model_dump()


def run_python_code(input_data: Dict) -> Dict:
    """Executes LLM-generated Python code directly. No sandbox."""
    try:
        inp = CodeInput.model_validate(input_data)
    except ValidationError as e:
        return {
            "status": "error",
            "error": f"CodeInput validation error: {e}",
            "stdout": None,
            "images": [],
            "result": None,
        }

    df = None
    if inp.csv_path:
        try:
            df = pd.read_csv(inp.csv_path)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load CSV: {e}",
                "stdout": None,
                "images": [],
                "result": None,
            }

    local_vars = {"df": df}

    buf = io.StringIO()
    error = None

    try:
        with contextlib.redirect_stdout(buf):
            exec(inp.code, {}, local_vars)
    except Exception:
        error = traceback.format_exc()

    stdout = buf.getvalue()

    # Save matplotlib figures
    images = []
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for idx, fig in enumerate(figs, start=1):
        os.makedirs(inp.work_dir, exist_ok=True)
        filename = f"{inp.session_id}_plot_{inp.iteration_count}_{idx}.png"
        path = os.path.join(inp.work_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        images.append(path)
    plt.close("all")

    result_obj = local_vars.get("result")

    return CodeOutput(
        status="success" if error is None else "error",
        stdout=stdout if stdout else None,
        error=error,
        images=images,
        result=result_obj
    ).model_dump()
