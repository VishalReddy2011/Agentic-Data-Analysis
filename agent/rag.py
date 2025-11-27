import os
from typing import List, Dict

# Base directory of the knowledge base
KB_ROOT = os.path.join("rag_kb")

def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _load_folder_texts(folder: str) -> List[str]:
    full_path = os.path.join(KB_ROOT, folder)
    out = []
    for name in os.listdir(full_path):
        fp = os.path.join(full_path, name)
        if os.path.isfile(fp):
            out.append(_load_text_file(fp))
    return out

# RAG-1: Statistical definitions
def load_stats_definitions() -> List[str]:
    return _load_folder_texts("stats")

# RAG-2: Style guide examples
def load_style_guide() -> List[str]:
    return _load_folder_texts("style")

# RAG-3: Interpretation guide
def load_interpretation_guide() -> List[str]:
    return _load_folder_texts("interpretation")

# Combined RAG dictionary for convenience
def load_all_rag() -> Dict[str, List[str]]:
    return {
        "stats": load_stats_definitions(),
        "style": load_style_guide(),
        "interpretation": load_interpretation_guide(),
    }
