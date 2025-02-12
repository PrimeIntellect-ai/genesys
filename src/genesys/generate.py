import os
import uuid
import sglang as sgl
from pydantic_config import parse_argv, BaseConfig
from pydantic import model_validator
from transformers import AutoTokenizer
from genesys.utils import (
    GcpBucket,
    display_config_panel,
    download_model,
    save_batch_results,
    generate_short_id,
    get_machine_info,
    log,
    console,
)
from genesys.data import DataConfig, DataLoaderGenesys
from genesys.toploc import build_proofs_base64, sha256sum


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

    pre_download_retry: int = 0

    @model_validator(mode="after")
    def check_batch_size(self):
        if self.sample_per_file < self.data.batch_size:
            raise ValueError("sample_per_file must be greater than or equal to batch_size")
        if self.data.max_samples is not None and self.data.max_samples < self.sample_per_file:
            raise ValueError("max_samples must be greater than or equal to sample_per_file")
        return self


def main(config: GenerateConfig):
    # Initial welcome table
    display_config_panel(console, config)

    log("[bold yellow] Loading model and initializing pipeline...[/]")

    # Initialize components
    log("[cyan] Configuring output path and gcp bucket...[/]")
    if not os.path.exists(config.path_output):
        os.makedirs(config.path_output)
    gcp_bucket = (
        GcpBucket(config.gcp_bucket, os.environ.get("GCP_CREDENTIALS_BASE64"))
        if config.gcp_bucket is not None
        else None
    )

    if config.pre_download_retry > 0:
        log("[cyan] Pre-downloading model...[/]")
        download_model(config.name_model, config.pre_download_retry)

    log("[cyan] Loading model and Engine...[/]")

    llm = sgl.Engine(
        model_path=config.name_model, tp_size=config.num_gpus, return_hidden_states=True, skip_tokenizer_init=True
    )

    log("[cyan] Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    log("[cyan] Loading dataloader...[/]")
    dataloader = DataLoaderGenesys(config.data, tokenizer=tokenizer, tokenize=True)
    machine_info = get_machine_info()

    log("[bold green]✨ Setup complete! Starting generation...[/]")

    # Rest of the generation logic
    sampling_params = dict(temperature=config.temperature, top_p=config.top_p, max_new_tokens=config.max_tokens)
    all_results = []
    total_samples = 0

    for batch_inputs, batch in dataloader:
        responses = llm.generate(input_ids=batch_inputs, sampling_params=sampling_params)
        for batch_input, batch_element, response in zip(batch_inputs, batch, responses):
            batch_element["llm_response"] = tokenizer.decode(response["token_ids"], skip_special_tokens=True)
            batch_element["response_id"] = f"{batch_element['problem_id']}_{generate_short_id()}"
            batch_element["model_name"] = config.name_model
            batch_element["generation_config"] = sampling_params
            batch_element["machine_info"] = machine_info
            batch_element["input_ids"] = batch_input
            batch_element["output_ids"] = response["token_ids"]
            batch_element["proof"] = "".join(
                build_proofs_base64(response["meta_info"]["hidden_states"], 32, 128, skip_prefill=True)
            )
            all_results.append(batch_element)
        total_samples += len(batch)

        if len(all_results) >= config.sample_per_file:
            file_name = f"{config.out_file_prefix}_{uuid.uuid4()}.jsonl"
            file = os.path.join(config.path_output, file_name)
            save_batch_results(all_results, file, gcp_bucket)
            file_sha = sha256sum(file)
            dataloader.prime_metric.log_prime({"file_sha": file_sha, "file_name": file_name})
            log(f"[bold green]✨ Saved {len(all_results)} samples to {file} with sha {file_sha or "NA"}[/]")
            all_results = []

    log(f"[bold green]✨ Generation complete! Total samples: {total_samples}[/]")


if __name__ == "__main__":
    config = GenerateConfig(**parse_argv())
    main(config)
