from pydantic import BaseModel
from typing import Dict, Optional


class Response(BaseModel):
    problem_id: str
    source: str
    task_type: str
    in_source_id: Optional[str]
    prompt: str
    gold_standard_solution: Optional[str]
    verification_info: Dict
    metadata: Dict
    llm_response: str  # llm response string
    response_id: str
    model_name: str
    generation_config: Dict  # sampling parameters
    machine_info: Dict | None
