from pydantic import model_validator
from pydantic_config import BaseConfig, parse_argv
import sglang as sgl
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from genesys.utils import GcpBucket, repeat_elements, save_batch_results, generate_short_id
import uuid

class Config(BaseConfig):
    name_model: str = "Qwen/QwQ-32B-Preview"
    num_responses_per_question: int = 1
    num_gpus: int = 8
    temperature: float = 0.6
    out_file_prefix: str = "out"
    max_tokens: int = 32_768
    batch_size: int = 10_000
    max_samples: int | None = None
    gcp_bucket: str | None = None  # optional, if provided, will save the each file with sample_per_file  to GCP
    sample_per_file: int = 10_000  # how much sample each file contains

    @model_validator(mode="after")
    def check_batch_size(self):
        if self.sample_per_file < self.batch_size:
            raise ValueError("sample_per_file must be greater than or equal to batch_size")
        if self.max_samples is not None and self.max_samples < self.sample_per_file:
            raise ValueError("max_samples must be greater than or equal to sample_per_file")
        return self


def main(config: Config):
    gcp_bucket = GcpBucket(config.gcp_bucket) if config.gcp_bucket is not None else None

    llm = sgl.Engine(model_path=config.name_model, tp_size=config.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    math_dataset = load_dataset("justus27/verifiable-tasks")["train"]
    math_dataset = math_dataset.add_column("problem_id", range(len(math_dataset))) # this should be part of the dataset and not added manually

    sampling_params = dict(temperature=config.temperature, max_new_tokens=config.max_tokens)
    max_samples = config.max_samples if config.max_samples is not None else len(math_dataset)

    all_results = []
    for i in tqdm(range(0, min(max_samples, len(math_dataset)), config.batch_size), desc="Generating data"):
        batch = math_dataset.select(list(range(i, min(i + config.batch_size, len(math_dataset)))))
        batch = repeat_elements([b for b in batch], config.num_responses_per_question) # turn hf dataset slice into list
        batch_messages = [[{"role": "user", "content": b["prompt"]}] for b in batch]

        batch_inputs = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        responses = llm.generate(batch_inputs, sampling_params)

        for i, (batch_element, response) in enumerate(zip(batch, responses)):
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
