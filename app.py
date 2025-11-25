# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import os
import threading
from io import BytesIO
from typing import Dict, Any

from agent.graph import app_agent
from agent.schemas import AgentInsight
from statistics import generate_descriptive_stats
from utils.pdf_generator import build_report


UPLOAD_DIR = "./uploads"
REPORT_DIR = "./reports"
IMAGE_DIR = "./static/images"
'''
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
'''
app = FastAPI(title="Data-Analysis Agent API")


def _cleanup_session_artifacts(session_id: str, csv_path: str, report_path: str) -> None:
    # Remove CSV
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    except Exception:
        pass

    # Remove generated report
    try:
        if os.path.exists(report_path):
            os.remove(report_path)
    except Exception:
        pass

    # Remove images associated with the session
    try:
        for name in os.listdir(IMAGE_DIR):
            if name.startswith(f"{session_id}_plot_") or name.startswith(f"{session_id}_"):
                try:
                    os.remove(os.path.join(IMAGE_DIR, name))
                except Exception:
                    pass
    except Exception:
        pass


def _run_agent_and_build_report(session_id: str, csv_path: str) -> None:
    """
    Background worker:
    - invoke LangGraph agent with the session + csv_path
    - compute descriptive stats
    - convert insights into AgentInsight objects
    - build PDF via build_report(session_id, descriptive_stats, list_of_insights)
    """
    try:
        # 1. Run the agent. The agent expects an initial state dict.
        initial_state: Dict[str, Any] = {"session_id": session_id, "csv_path": csv_path}
        result = app_agent.invoke(initial_state)

        # 2. Generate descriptive stats object
        descriptive_stats = generate_descriptive_stats(csv_path)

        # 3. Convert final_insights (dicts) to AgentInsight models where possible
        final_insights_raw = result.get("final_insights", []) or []
        insights = []
        for item in final_insights_raw:
            try:
                # If item is already a dict with insight_text, technique_used, graph_filename
                # validate/parse into AgentInsight
                ins = AgentInsight.model_validate(item)
                insights.append(ins)
            except Exception:
                # Fallback: craft a minimal AgentInsight from raw text/fields
                try:
                    text = item.get("insight_text") if isinstance(item, dict) else str(item)
                except Exception:
                    text = str(item)
                try:
                    technique = item.get("technique_used") if isinstance(item, dict) else ""
                except Exception:
                    technique = ""
                insights.append(AgentInsight(insight_text=str(text), technique_used=str(technique), graph_filename=None))

        # 4. Build the PDF (returns doc path)
        doc_path = build_report(session_id, descriptive_stats, insights, output_dir=REPORT_DIR)

        # No return value — report is written to REPORT_DIR/{session_id}.pdf
        return
    except Exception:
        # Log error to a local file for debugging (do not crash background thread)
        try:
            with open(os.path.join(REPORT_DIR, f"{session_id}_error.log"), "w", encoding="utf-8") as fh:
                import traceback
                fh.write(traceback.format_exc())
        except Exception:
            pass
        return


@app.post("/upload", status_code=202)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV endpoint.
    - saves CSV to uploads/{session_id}.csv
    - spawns background thread to run the agent and produce PDF
    - returns {session_id, status: "processing"}
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    session_id = str(uuid.uuid4())
    filename = f"{session_id}.csv"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save uploaded file
    try:
        with open(file_path, "wb") as out_f:
            content = await file.read()
            out_f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Spawn background thread to run the agent and build the report
    thread = threading.Thread(target=_run_agent_and_build_report, args=(session_id, file_path), daemon=True)
    thread.start()

    return JSONResponse(status_code=202, content={"session_id": session_id, "status": "processing"})


@app.get("/report")
def get_report(session_id: str):
    """
    Polling endpoint to retrieve the final PDF.
    - If PDF is ready: load into memory, delete artifacts, and return PDF bytes.
    - If not ready: return {status: "processing"}.
    """
    report_filename = f"{session_id}.pdf"
    report_path = os.path.join(REPORT_DIR, report_filename)

    if not os.path.exists(report_path):
        # Report not ready yet
        return JSONResponse(status_code=202, content={"status": "processing"})

    # Read PDF into memory, delete artifacts, then stream to user
    try:
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {e}")

    # Cleanup all session artifacts (CSV, images, report)
    csv_path = os.path.join(UPLOAD_DIR, f"{session_id}.csv")
    _cleanup_session_artifacts(session_id, csv_path, report_path)

    # Return PDF bytes (in-memory) and appropriate headers
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="{report_filename}"'
    })
