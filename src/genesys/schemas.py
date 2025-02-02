from pydantic import BaseModel
from typing import Dict, Optional


class UnscoredResult(BaseModel):
    problem_id: str
    source: str
    task_type: str
    in_source: Optional[str]
    prompt: str
    gold_standard_solution: Optional[str]
    verification_info: Dict
    metadata: Dict
    llm_response: str
    response_id: str
    model_name: str
    generation_config: Dict
