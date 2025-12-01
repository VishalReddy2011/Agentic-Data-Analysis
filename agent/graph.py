from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import subprocess

from agent.schemas import AgentInsight
from agent.tools import get_csv_profile, run_python_code
from agent.rag import load_style_guide

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

use_openai = True
reason_model = "mistral:instruct"
vision_model = "moondream:1.8b"

models = { "reason": reason_model, "vision": vision_model }

class AgentState(BaseModel):
    session_id: str
    csv_path: str
    column_profile: Dict[str, Any] = Field(default_factory=dict)
    final_insights: List[AgentInsight] = Field(default_factory=list)
    current_plan: str = ""
    plan_history: List[str] = Field(default_factory=list)
    current_tool_output: Dict[str, Any] = None
    insight: Optional[AgentInsight] = None

def _kill_ollama_model():
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    lines = result.stdout.splitlines()
    if len(lines) > 1:
        model_name = lines[1].split()[0]
        subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            text=True
        )

def _get_current_ollama_model():
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    lines = result.stdout.splitlines()
    if len(lines) > 1:
        model_name = lines[1].split()[0]
        return model_name
    return None

def _create_model(type: str) -> ChatOllama:
    if use_openai:
        return ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
    if _get_current_ollama_model() != models[type]:
        _kill_ollama_model()
    return ChatOllama(model=models[type], temperature=0)

def profile_data_node(state: AgentState) -> Dict[str, Any]:
    out = get_csv_profile(state.csv_path)
    return {"column_profile": out}


def plan_analysis_node(state: AgentState) -> Dict[str, Any]:
    llm = _create_model(type="reason")

    plan_prompt = f"""
You are an expert data analyst. Output ONE plan command ONLY.

CSV Profile:
{state.column_profile}

Previous plans:
{" ,".join(state.plan_history) if state.plan_history else 'None'}

Available commands:
- correlate colA colB
- histogram col
- value_counts col
- finish

Rules:
- Never repeat a previous plan.
- If nothing left to analyze, return "finish".
- Answer in plain text, no formatting.

Return only the command in the exact given format in 'Available commands'.
"""

    resp = llm.invoke(plan_prompt)
    plan = resp.content.strip().splitlines()[0]
    return {"current_plan": plan, "plan_history": state.plan_history + [plan]}


def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    plan = state.current_plan.strip().lower()

    if plan == "finish":
        return {"current_tool_output": None}

    tokens = plan.split()
    cmd = tokens[0]

    csv_path = state.csv_path
    session_id = state.session_id
    iteration = len(state.final_insights)

    if cmd == "correlate" and len(tokens) >= 3:
        a, b = tokens[1], tokens[2]
        code = (
            "from scipy import stats\n"
            f"a = df['{a}'].dropna().astype(float)\n"
            f"b = df['{b}'].dropna().astype(float)\n"
            "corr, p = stats.pearsonr(a, b)\n"
            "result = {'type': 'numeric', 'correlation': float(corr), 'pvalue': float(p)}\n"
        )

    elif cmd == "value_counts" and len(tokens) >= 2:
        col = tokens[1]
        code = (
            f"vc = df['{col}'].value_counts().head(20).to_dict()\n"
            "result = {'type': 'numeric', 'value_counts': vc}\n"
        )

    elif cmd == "histogram" and len(tokens) >= 2:
        col = tokens[1]
        code = (
            f"vals = df['{col}'].dropna().astype(float)\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            f"plt.hist(vals, bins=30)\n"
            f"plt.title('Histogram of {col}')\n"
            "result = {'type': 'image'}\n"
        )

    else:
        raise ValueError(f"Unknown command: {plan}")

    payload = {
        "code": code,
        "csv_path": csv_path,
        "session_id": session_id,
        "iteration_count": iteration,
    }

    out = run_python_code(payload)
    return {"current_tool_output": out}


def generate_insight_node(state: AgentState) -> Dict[str, Any]:
    output = state.current_tool_output

    rag_style = "\n".join(load_style_guide())

    if output is None:
        return {"insight": None}

    tool_result = output.get("result")
    is_image = isinstance(tool_result, dict) and tool_result.get("type") == "image"

    if is_image:
        img_path = tool_result["image_path"]
        llm = _create_model(type="vision").with_structured_output(schema = AgentInsight)

        prompt = f"""
PLAN: {state.current_plan}
A graph image is provided. Analyze it and produce one concise, professional insight in one sentence. Do not use any markdown formatting. Do not mention anything else.

Styling rules:
{rag_style}
"""

        resp = llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": img_path}
                ]
            }
        ])

        return {"insight": resp}

    else:
        llm = _create_model(type="reason").with_structured_output(schema = AgentInsight)

        prompt = f"""
PLAN: {state.current_plan}

Numeric Result:
{tool_result}

Styling rules:
{rag_style}
"""

        resp = llm.invoke(prompt)
        return {"insight": resp}



def add_to_final_list_node(state: AgentState) -> Dict[str, Any]:
    fin = list(state.final_insights)
    if state.insight:
        fin.append(state.insight)
    return {"final_insights": fin}


graph = StateGraph(AgentState)

graph.add_node("profile_data", profile_data_node)
graph.add_node("plan_analysis", plan_analysis_node)
graph.add_node("execute_tool", execute_tool_node)
graph.add_node("generate_insight", generate_insight_node)
graph.add_node("add_to_final_list", add_to_final_list_node)

graph.set_entry_point("profile_data")


def should_finish(state: AgentState) -> bool:
    if len(state.final_insights) >= 5:
        return True
    return (state.current_plan or "").strip().lower() == "finish"


graph.add_edge("profile_data", "plan_analysis")
graph.add_conditional_edges("plan_analysis", should_finish, {True: END, False: "execute_tool"})
graph.add_edge("execute_tool", "generate_insight")
graph.add_edge("generate_insight", "add_to_final_list")
graph.add_edge("add_to_final_list", "plan_analysis")

app_agent = graph.compile()
