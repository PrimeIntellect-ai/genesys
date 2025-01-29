from pydantic import model_validator
from pydantic_config import BaseConfig, parse_argv
from genesys.data import DataLoaderGenesys, DataConfig
import sglang as sgl
from transformers import AutoTokenizer
from genesys.utils import GcpBucket, save_batch_results, generate_short_id
import uuid


class Config(BaseConfig):
    name_model: str = "Qwen/QwQ-32B-Preview"
    num_gpus: int = 8
    out_file_prefix: str = "out"
    max_tokens: int = 32_768
    temperature: float = 0.9

    gcp_bucket: str | None = None  # optional, if provided, will save the each file with sample_per_file  to GCP
    sample_per_file: int = 10_000  # how much sample each file contains

    data: DataConfig = DataConfig()

    @model_validator(mode="after")
    def check_batch_size(self):
        if self.sample_per_file < self.data.batch_size:
            raise ValueError("sample_per_file must be greater than or equal to batch_size")
        if self.data.max_samples is not None and self.data.max_samples < self.sample_per_file:
            raise ValueError("max_samples must be greater than or equal to sample_per_file")
        return self


def main(config: Config):
    gcp_bucket = GcpBucket(config.gcp_bucket) if config.gcp_bucket is not None else None

    llm = sgl.Engine(model_path=config.name_model, tp_size=config.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    dataloader = DataLoaderGenesys(config.data, tokenizer=tokenizer)

    sampling_params = dict(temperature=config.temperature, max_new_tokens=8192, stop=["<|eot_id|>"])

    all_results = []

    for batch_inputs, batch in dataloader:
        responses = llm.generate(batch_inputs, sampling_params)

        for batch_element, response in zip(batch, responses):
            batch_element["llm_response"] = response["text"]
            batch_element["response_id"] = f"{batch_element['problem_id']}_{generate_short_id()}"
            batch_element["model_name"] = config.name_model
            batch_element["generation_config"] = dict(temperature=config.temperature)

            all_results.append(batch_element)

        if len(all_results) >= config.sample_per_file:
            file_name = f"{config.out_file_prefix}_{uuid.uuid4()}.jsonl"
            save_batch_results(all_results, file_name, gcp_bucket)
            all_results = []


if __name__ == "__main__":
    config = Config(**parse_argv())
    main(config)
