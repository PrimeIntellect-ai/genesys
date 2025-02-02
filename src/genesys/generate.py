import os
import uuid
import sglang as sgl
from pydantic_config import parse_argv
from transformers import AutoTokenizer
from rich.console import Console
from genesys.utils import GcpBucket, display_config_panel, save_batch_results, generate_short_id
from genesys.data import DataLoaderGenesys
from genesys.schemas import GenerateConfig


def main(config: GenerateConfig):
    console = Console()

    # Initial welcome table
    display_config_panel(console, config)

    # Loading message
    console.print("\n[bold yellow] Loading model and initializing pipeline...[/]\n")

    # Initialize components
    if not os.path.exists(config.path_output):
        os.makedirs(config.path_output)
    gcp_bucket = (
        GcpBucket(config.gcp_bucket, os.environ.get("GCP_CREDENTIALS_BASE64"))
        if config.gcp_bucket is not None
        else None
    )
    llm = sgl.Engine(model_path=config.name_model, tp_size=config.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(config.name_model)
    dataloader = DataLoaderGenesys(config.data, tokenizer=tokenizer)

    # Ready message
    console.print("[bold green]✨ Setup complete! Starting generation...\n[/]")

    # Rest of the generation logic
    sampling_params = dict(temperature=config.temperature, top_p=config.top_p, max_new_tokens=config.max_tokens)
    all_results = []
    total_samples = 0

    for batch_inputs, batch in dataloader:
        responses = llm.generate(batch_inputs, sampling_params)
        for batch_element, response in zip(batch, responses):
            batch_element["llm_response"] = response["text"]
            batch_element["response_id"] = f"{batch_element['problem_id']}_{generate_short_id()}"
            batch_element["model_name"] = config.name_model
            batch_element["generation_config"] = sampling_params
            all_results.append(batch_element)
        total_samples += len(batch)

        if len(all_results) >= config.sample_per_file:
            file_name = f"{config.out_file_prefix}_{uuid.uuid4()}.jsonl"
            file = os.path.join(config.path_output, file_name)
            save_batch_results(all_results, file, gcp_bucket)
            all_results = []

    console.print(f"[bold green]✨ Generation complete! Total samples: {total_samples}[/]")


if __name__ == "__main__":
    config = GenerateConfig(**parse_argv())
    main(config)
