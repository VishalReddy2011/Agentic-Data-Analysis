from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import os

from agent.schemas import AgentInsight
from agent.tools import get_csv_profile, run_python_code
from agent.rag import (
    load_stats_definitions,
    load_style_guide,
    load_interpretation_guide,
    load_matplotlib_examples
)

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class AgentState(BaseModel):
    session_id: str
    csv_path: str
    column_profile: Dict[str, Any] = Field(default_factory=dict)
    final_insights: List[AgentInsight] = Field(default_factory=list)
    agent_scratchpad: List[Any] = Field(default_factory=list)
    current_plan: str = ""
    current_tool_output: Any = None
    visualization_code: str = ""
    iteration_count: int = 0
    insight: Optional[AgentInsight] = None


def create_model() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# GRAPH NODES

def profile_data_node(state: AgentState) -> Dict[str, Any]:
    out = get_csv_profile({"file_path": state.csv_path})
    return {"column_profile": out}


def plan_analysis_node(state: AgentState) -> Dict[str, Any]:
    llm = create_model()

    rag_stats = "\n".join(load_stats_definitions())
    rag_interpret = "\n".join(load_interpretation_guide())

    prompt = f"""
You are an expert data analyst.
You must output ONE plan command ONLY.

CSV Profile:
{state.column_profile}

Available commands:
- correlate colA colB
- histogram col
- value_counts col
- finish

Rules:
- If enough analysis done or nothing left → return "finish".
- Otherwise choose the next valid command.

Statistical knowledge:
{rag_stats}

Interpretation rules:
{rag_interpret}

Return only one line: the exact command.
"""

    plan = llm.invoke(prompt).content.strip().splitlines()[0]
    return {"current_plan": plan}


def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    plan = state.current_plan
    tokens = plan.split()
    cmd = tokens[0].lower()

    session_id = state.session_id
    iteration = state.iteration_count
    csv_path = state.csv_path

    if cmd == "correlate" and len(tokens) >= 3:
        a, b = tokens[1], tokens[2]
        code = (
            "from scipy import stats\n"
            f"a = df['{a}'].dropna().astype(float)\n"
            f"b = df['{b}'].dropna().astype(float)\n"
            "corr, p = stats.pearsonr(a, b)\n"
            "result = {'correlation': float(corr), 'pvalue': float(p)}\n"
            "print('correlation:', corr)\n"
            "print('pvalue:', p)\n"
        )

    elif cmd == "histogram" and len(tokens) >= 2:
        col = tokens[1]
        code = (
            f"vals = df['{col}'].dropna().astype(float)\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            f"plt.hist(vals, bins=30)\n"
            f"plt.title('Histogram of {col}')\n"
            "path = f'static/images/{session_id}_plot_{iteration}_hist.png'\n"
            "plt.savefig(path, bbox_inches='tight')\n"
            "saved_images = [path]\n"
        )

    elif cmd == "value_counts" and len(tokens) >= 2:
        col = tokens[1]
        code = (
            f"vc = df['{col}'].value_counts().head(20).to_dict()\n"
            "result = vc\n"
            "print('value_counts:', vc)\n"
        )

    else:
        raise ValueError(f"Unknown plan command: {plan}")

    payload = {
        "code": code,
        "csv_path": csv_path,
        "session_id": session_id,
        "iteration_count": iteration,
        "work_dir": "static/images"
    }

    out = run_python_code(payload)
    return {"current_tool_output": out}


def generate_insight_node(state: AgentState) -> Dict[str, Any]:
    llm = create_model().with_structured_output(schema=AgentInsight)

    rag_style = "\n".join(load_style_guide())
    rag_interpret = "\n".join(load_interpretation_guide())
    rag_stats = "\n".join(load_stats_definitions())

    prompt = f"""
You are a senior data analyst.

PLAN: {state.current_plan}
TOOL OUTPUT: {state.current_tool_output}

Your tasks:
1. Interpret the statistical results accurately.
2. Produce a concise professional insight.
3. Follow the style rules below.

Style Guide:
{rag_style}

Interpretation Guide:
{rag_interpret}

Statistical Definitions:
{rag_stats}

Return ONLY a valid AgentInsight object.
"""

    insight = llm.invoke(prompt)
    return {"insight": insight}


def generate_visualization_node(state: AgentState) -> Dict[str, Any]:
    llm = create_model()

    rag_matplotlib = "\n".join(load_matplotlib_examples())

    prompt = f"""
You must write Python matplotlib code to visualize the following insight:

INSIGHT:
{state.insight}

CSV columns can be accessed via df['col_name'].

Rules:
- Produce exactly one visualization.
- Use a clean, minimal style.
- Use the examples for reference.
- Save the figure to:
  static/images/{state.session_id}_plot_{state.iteration_count}_viz.png

Matplotlib Examples:
{rag_matplotlib}

Return ONLY valid Python code. No explanation.
"""

    code = llm.invoke(prompt).content
    return {"visualization_code": code}


def execute_visualization_node(state: AgentState) -> Dict[str, Any]:
    payload = {
        "code": state.visualization_code,
        "csv_path": state.csv_path,
        "session_id": state.session_id,
        "iteration_count": state.iteration_count,
        "work_dir": "static/images"
    }
    out = run_python_code(payload)
    return {"visualization_out": out}


def add_to_final_list_node(state: AgentState) -> Dict[str, Any]:
    fin = list(state.final_insights)
    fin.append(state.insight)
    return {
        "final_insights": fin,
        "iteration_count": state.iteration_count + 1
    }


# GRAPH BUILD

graph = StateGraph(AgentState)

graph.add_node("profile_data", profile_data_node)
graph.add_node("plan_analysis", plan_analysis_node)
graph.add_node("execute_tool", execute_tool_node)
graph.add_node("generate_insight", generate_insight_node)
graph.add_node("generate_visualization", generate_visualization_node)
graph.add_node("execute_visualization", execute_visualization_node)
graph.add_node("add_to_final_list", add_to_final_list_node)

graph.set_entry_point("profile_data")


def should_finish(state: AgentState) -> bool:
    return (state.current_plan or "").strip().lower() == "finish"


graph.add_edge("profile_data", "plan_analysis")
graph.add_conditional_edges("plan_analysis", should_finish, {True: END, False: "execute_tool"})
graph.add_edge("execute_tool", "generate_insight")
graph.add_edge("generate_insight", "generate_visualization")
graph.add_edge("generate_visualization", "execute_visualization")
graph.add_edge("execute_visualization", "add_to_final_list")
graph.add_edge("add_to_final_list", "plan_analysis")

app_agent = graph.compile()
