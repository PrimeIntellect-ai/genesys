import os
from pydantic import model_validator
from pydantic_config import BaseConfig, parse_argv
from genesys.data import DataLoaderGenesys, DataConfig
import sglang as sgl
from transformers import AutoTokenizer
from genesys.utils import GcpBucket, display_config_panel, save_batch_results, generate_short_id
import uuid

from rich.console import Console


class Config(BaseConfig):
    name_model: str = "Qwen/QwQ-32B-Preview"
    num_gpus: int = 8
    max_tokens: int = 32_768
    temperature: float = 0.9

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


def main(config: Config):
    console = Console()

    # Initial welcome table
    display_config_panel(console, config)

    # Loading message
    console.print("\n[bold yellow] Loading model and initializing pipeline...[/]\n")

    # Initialize components
    if not os.path.exists(config.path_output):
        os.makedirs(config.path_output)
    gcp_bucket = GcpBucket(config.gcp_bucket) if config.gcp_bucket is not None else None
    llm = sgl.Engine(model_path=config.name_model, tp_size=config.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)
    dataloader = DataLoaderGenesys(config.data, tokenizer=tokenizer)

    # Ready message
    console.print("[bold green]✨ Setup complete! Starting generation...\n[/]")

    # Rest of the generation logic
    sampling_params = dict(temperature=config.temperature, max_new_tokens=8192, stop=["<|eot_id|>"])
    all_results = []
    total_samples = 0

    for batch_inputs, batch in dataloader:
        responses = llm.generate(batch_inputs, sampling_params)
        for batch_element, response in zip(batch, responses):
            batch_element["llm_response"] = response["text"]
            batch_element["response_id"] = f"{batch_element['problem_id']}_{generate_short_id()}"
            batch_element["model_name"] = config.name_model
            batch_element["generation_config"] = dict(temperature=config.temperature)
            all_results.append(batch_element)
        total_samples += len(batch)

        if len(all_results) >= config.sample_per_file:
            file_name = f"{config.out_file_prefix}_{uuid.uuid4()}.jsonl"
            file = os.path.join(config.path_output, file_name)
            save_batch_results(all_results, file, gcp_bucket)
            all_results = []

    console.print(f"[bold green]✨ Generation complete! Total samples: {total_samples}[/]")


if __name__ == "__main__":
    config = Config(**parse_argv())
    main(config)
