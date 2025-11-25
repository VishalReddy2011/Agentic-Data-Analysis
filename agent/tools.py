from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import io
import os
import traceback
import base64
import uuid
import contextlib

# Pydantic models

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
    status: str = Field(description="success | error")
    stdout: Optional[str] = None
    error: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    result: Optional[Any] = None

    model_config = {"extra": "forbid", "validate_assignment": True}


# Helper

def _capture_exec(code: str, safe_globals: Dict[str, Any], safe_locals: Dict[str, Any]):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals, safe_locals)
        stdout = buf.getvalue()
        return {"stdout": stdout, "locals": safe_locals, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        return {"stdout": buf.getvalue(), "locals": safe_locals, "error": tb}

# Tool: get_csv_profile

def get_csv_profile(input_data: Dict) -> Dict:
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
            unique_count = int(ser.nunique(dropna=True))
        elif pd.api.types.is_datetime64_any_dtype(ser):
            dtype = "datetime"
            unique_count = int(ser.nunique(dropna=True))
        else:
            dtype = "categorical"
            unique_count = int(ser.nunique(dropna=True))
        sample = ser.dropna().astype(str).head(5).tolist()
        columns[str(col)] = ColumnProfile(
            name=str(col),
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

# Tool: run_python_code

def run_python_code(input_data: Dict) -> Dict:
    try:
        inp = CodeInput.model_validate(input_data)
    except ValidationError as e:
        return {"status": "error", "error": f"CodeInput validation error: {e}", "stdout": None, "images": [], "result": None}

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        },
        "pd": pd,
        "np": np,
        "plt": plt,
        "stats": stats,
    }

    safe_locals: Dict[str, Any] = {}

    if inp.csv_path:
        try:
            df = pd.read_csv(inp.csv_path)
            safe_globals["df"] = df
            safe_locals["df"] = df
        except Exception as e:
            return {"status": "error", "error": f"Failed to load CSV: {e}", "stdout": None, "images": [], "result": None}

    unique_prefix = inp.session_id or str(uuid.uuid4())
    iteration = inp.iteration_count or 0

    exec_result = _capture_exec(inp.code, safe_globals, safe_locals)

    stdout = exec_result["stdout"]
    error = exec_result["error"]
    locals_after = exec_result["locals"]

    images: List[str] = []

    # Detect common matplotlib saves in locals or created figures
    # If user code set a variable 'saved_images' with paths, respect it.
    if "saved_images" in locals_after:
        try:
            for p in locals_after["saved_images"]:
                if isinstance(p, str) and os.path.exists(p):
                    images.append(p)
        except Exception:
            pass

    # Also check for open figures and save them automatically
    try:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for idx, fig in enumerate(figs, start=1):
            img_name = f"{unique_prefix}_plot_{iteration}_{idx}.png"
            img_path = os.path.join(inp.work_dir, img_name)
            fig.savefig(img_path, bbox_inches="tight")
            images.append(img_path)
        plt.close("all")
    except Exception:
        pass

    result_obj = None
    if "result" in locals_after:
        try:
            result_obj = locals_after["result"]
        except Exception:
            result_obj = None

    out = CodeOutput(
        status="success" if error is None else "error",
        stdout=stdout if stdout else None,
        error=error,
        images=images,
        result=result_obj
    )
    return out.model_dump()
