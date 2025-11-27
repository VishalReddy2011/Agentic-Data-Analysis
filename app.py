from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import os
import threading
from io import BytesIO
from typing import Dict, Any

from agent.graph import app_agent
from langchain_core.runnables.config import RunnableConfig
from agent.schemas import AgentInsight
from utils.statistics import generate_descriptive_stats
from utils.pdf_generator import build_report

from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = "./uploads"
REPORT_DIR = "./reports"
IMAGE_DIR = "./static/images"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI(title="Data-Analysis Agent API")


def _cleanup_session_artifacts(session_id: str, csv_path: str, report_path: str) -> None:
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    except Exception:
        pass

    try:
        if os.path.exists(report_path):
            os.remove(report_path)
    except Exception:
        pass

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
    try:
        initial_state: Dict[str, Any] = {"session_id": session_id, "csv_path": csv_path}
        result = app_agent.invoke(initial_state, config=RunnableConfig(recursion_limit=100))

        descriptive_stats = generate_descriptive_stats(csv_path)

        final_insights_raw = result.get("final_insights", []) or []
        insights = []
        for item in final_insights_raw:
            try:
                if isinstance(item, AgentInsight):
                    insights.append(item)
                    continue
                if isinstance(item, dict):
                    ins = AgentInsight.model_validate(item)
                    insights.append(ins)
                    continue
                ins = AgentInsight.model_validate(item)
                insights.append(ins)
            except Exception:
                try:
                    text = item.get("insight_text") if isinstance(item, dict) else str(item)
                except Exception:
                    text = str(item)
                try:
                    technique = item.get("technique_used") if isinstance(item, dict) else ""
                except Exception:
                    technique = ""
                graph_file = item.get("graph_filename") if isinstance(item, dict) else None

                insights.append(AgentInsight(
                    insight_text=str(text),
                    technique_used=str(technique),
                    graph_filename=graph_file
                ))

        doc_path = build_report(session_id, descriptive_stats, insights, output_dir=REPORT_DIR)
        return

    except Exception:
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(os.path.join(REPORT_DIR, f"{session_id}_error.log"), "w", encoding="utf-8") as fh:
                import traceback
                fh.write(traceback.format_exc())
        except Exception:
            pass
        return


@app.post("/upload", status_code=202)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    session_id = str(uuid.uuid4())
    filename = f"{session_id}.csv"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as out_f:
            content = await file.read()
            out_f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    thread = threading.Thread(target=_run_agent_and_build_report, args=(session_id, file_path), daemon=True)
    thread.start()

    return JSONResponse(status_code=202, content={"session_id": session_id, "status": "processing"})


@app.get("/report")
def get_report(session_id: str):
    report_filename = f"{session_id}.pdf"
    report_path = os.path.join(REPORT_DIR, report_filename)

    error_log_path = os.path.join(REPORT_DIR, f"{session_id}_error.log")

    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()

        csv_path = os.path.join(UPLOAD_DIR, f"{session_id}.csv")

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{report_filename}"'
            }
        )

    if os.path.exists(error_log_path):
        with open(error_log_path, "r") as f:
            error_text = f.read()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Agent failed during processing.",
                "traceback": error_text[-5000:],
            },
        )

    return JSONResponse(status_code=202, content={"status": "processing"})
