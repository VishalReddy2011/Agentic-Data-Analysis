from pydantic import BaseModel, Field
from typing import Optional

class AgentInsight(BaseModel):
    insight_text: str = Field(
        description="Concise human-readable insight containing all numerical results."
    )

    technique_used: str = Field(
        description="Statistical technique used, e.g., Pearson Correlation."
    )
    
    model_config = {
        "extra": "forbid",
        "validate_assignment": True
    }