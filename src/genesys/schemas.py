from pydantic_config import BaseConfig
from pydantic import model_validator
from typing import Dict, Optional
from genesys.data import DataConfig


class GenerateConfig(BaseConfig):
    name_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    num_gpus: int = 8
    max_tokens: int = 32_768
    temperature: float = 0.6
    top_p: float = 0.95

    data: DataConfig = DataConfig()

    # output file
    out_file_prefix: str = "out"
    gcp_bucket: str | None = None  # optional, if provided, will save the each file with sample_per_file  to GCP
    sample_per_file: int = 10_000  # how much sample each file contains
    path_output: str = "output"

    @model_validator(mode="after")
    def check_batch_size(self):
        if self.sample_per_file < self.data.batch_size:
            raise ValueError("sample_per_file must be greater than or equal to batch_size")
        if self.data.max_samples is not None and self.data.max_samples < self.sample_per_file:
            raise ValueError("max_samples must be greater than or equal to sample_per_file")
        return self


class VerifyConfig(BaseConfig):
    file: str


class UnscoredResult(BaseConfig):
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
