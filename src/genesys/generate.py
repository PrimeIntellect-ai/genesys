import itertools
from pydantic import model_validator
from pydantic_config import BaseConfig, parse_argv
import torch
import sglang as sgl
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from genesys.utils import GcpBucket, repeat_elements, save_batch_results
import uuid

SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with: \n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem. If the question is a multiple choice question, [answer] should be the letter indicating your correct response (e.g. \\text{A} or \\text{B})."


class Config(BaseConfig):
    name_model: str = "Qwen/QwQ-32B-Preview"
    num_responses_per_question: int = 1
    num_gpus: int = 8
    temperature: float = 0.9
    batch_size: int = 10_000
    max_samples: int | None = None
    gcp_bucket: str | None = None  # optional, if provided, will save the each file with sample_per_file  to GCP
    sample_per_file: int = 10_000  # how much sample each file contains

    top_hidden_states_num: int = -1  # -1 mean all value

    @model_validator(mode="after")
    def check_batch_size(self):
        if self.sample_per_file < self.batch_size:
            raise ValueError("sample_per_file must be greater than or equal to batch_size")
        if self.max_samples is not None and self.max_samples < self.sample_per_file:
            raise ValueError("max_samples must be greater than or equal to sample_per_file")
        return self


def main(config: Config):
    gcp_bucket = GcpBucket(config.gcp_bucket) if config.gcp_bucket is not None else None

    llm = sgl.Engine(
        model_path=config.name_model, tp_size=config.num_gpus, top_hidden_states_nums=config.top_hidden_states_num
    )
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    math_dataset = load_dataset("PrimeIntellect/NuminaMath-groundtruth")["train"]
    math_dataset = math_dataset.add_column("problem_id", range(len(math_dataset)))

    sampling_params = dict(temperature=config.temperature, max_new_tokens=8192, stop=["<|eot_id|>"])

    max_samples = config.max_samples if config.max_samples is not None else len(math_dataset)

    all_results = []

    for i in tqdm(range(0, min(max_samples, len(math_dataset)), config.batch_size), desc="Generating data"):
        batch = math_dataset[i : min(i + config.batch_size, len(math_dataset))]
        batch_ids = list(
            itertools.chain.from_iterable([[idx] * config.num_responses_per_question for idx in batch["problem_id"]])
        )
        batch_ground_truths = list(
            itertools.chain.from_iterable([[gt] * config.num_responses_per_question for gt in batch["ground_truth"]])
        )

        batch_messages = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
            for problem in batch["problem"]
        ]
        batch_messages = repeat_elements(batch_messages, config.num_responses_per_question)
        batch_inputs = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        batch_output = llm.generate(batch_inputs, sampling_params, return_hidden_states=True)

        for j, out in enumerate(batch_output):
            result = dict()
            result["prompt"] = batch_messages[j][1]["content"]
            result["response"] = out["text"]
            result["problem_id"] = int(batch_ids[j])
            result["ground_truth"] = batch_ground_truths[j]

            hidden_states_val = torch.Tensor(out["meta_info"]["output_top_hidden_states_val"])

            if "output_top_hidden_states_idx" in out["meta_info"]:
                hidden_states_idx = torch.Tensor(out["meta_info"]["output_top_hidden_states_idx"])
            print(f"{hidden_states_val.shape=}, {hidden_states_idx.shape=}")

            all_results.append(result)

        if len(all_results) >= config.sample_per_file:
            file_name = f"out_{uuid.uuid4()}.jsonl"
            save_batch_results(all_results, file_name, gcp_bucket)
            all_results = []


if __name__ == "__main__":
    config = Config(**parse_argv())
    main(config)
