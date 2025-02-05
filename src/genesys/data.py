from typing import Generator
from pydantic_config import BaseConfig
from datasets import load_dataset
import rich.progress
from transformers import AutoTokenizer
import random
from genesys.prime_metrics import PrimeMetric


class DataConfig(BaseConfig):
    path: str = "PrimeIntellect/verifiable-tasks"
    ratio: list[float] | None = None
    max_samples: int | None = None
    batch_size: int = 10_000
    num_responses_per_question: int = 1

    shuffle: bool = True

    prime_log: bool = False

    prime_log_freq: int = 5


def repeat_elements(lst, n):
    return [item for item in lst for _ in range(n)]


class DataLoaderGenesys:
    """
    This is a naive dataloader that take multiple datasets and sample from them in a round-robin fashion.
    It allow to pass probability (called ratio here) to sample from the datasets.

    The round robin is done at the batch level, each batch only contain data from the same dataset.
    once a dataset is exhausted, it is not sampled from anymore.

    The loading could be done async with a backgroud process, but the performance are not critical yet hence it kept simple.

    Each dataset that is pass must have a "train" split and the content must be a list of dict with at least a "problem" and a "ground_truth" key.
    """

    def __init__(self, config: DataConfig, tokenizer: AutoTokenizer):
        self.config = config

        self.paths = list(config.path.split(","))

        datasets = [load_dataset(path)["train"] for path in self.paths]

        if config.shuffle:
            datasets = [data.shuffle() for data in datasets]

        if config.ratio is not None:
            ratio = [float(r) for r in config.ratio.split(",")]
            assert len(ratio) == len(datasets), "Number of paths and ratios must be the same"
        else:
            ratio = [1.0] * len(datasets)

        self.normalized_ratios = [r / sum(ratio) for r in ratio]

        def _add_column(dataset, path):
            dataset = dataset.add_column("hf_dataset_name", [path] * len(dataset))
            return dataset

        self.datasets = [_add_column(dataset, path) for dataset, path in zip(datasets, self.paths)]

        total_samples = sum(len(dataset) for dataset in datasets)
        max_samples = config.max_samples if config.max_samples is not None else total_samples

        self.total_samples = min(max_samples, total_samples)

        self.tokenizer = tokenizer

        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

        self.progress_bars = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.SpinnerColumn(),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TextColumn("samples |"),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("remaining "),
        )

        self.tasks = [
            self.progress_bars.add_task(f"{self.paths[i].split('/')[-1]}", total=length)
            for i, length in enumerate(self.dataset_lengths)
        ]

        self.prime_metric = PrimeMetric(disable=not (config.prime_log), period=config.prime_log_freq)

    def _prepare_batch(self, batch: dict, dataset: str) -> tuple:
        batch = repeat_elements(
            [b for b in batch], self.config.num_responses_per_question
        )  # turn hf dataset slice into list
        batch_messages = [[{"role": "user", "content": b["prompt"]}, {"role": "assistant", "content": "<think>/n"}] for b in batch]

        batch_inputs = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, continue_final_message=True)

        return batch_inputs, batch

    def __iter__(self) -> Generator[tuple, None, None]:
        with self.progress_bars:
            datasets_sample_counter = [0] * len(self.datasets)
            idx = 0

            while True:
                # Use weighted random choice instead of simple round robin
                current_dataset_index = random.choices(range(len(self.datasets)), weights=self.normalized_ratios, k=1)[
                    0
                ]

                current_dataset = self.datasets[current_dataset_index]

                if datasets_sample_counter[current_dataset_index] >= self.dataset_lengths[current_dataset_index]:
                    continue

                start = datasets_sample_counter[current_dataset_index]
                end = min(start + self.config.batch_size, self.dataset_lengths[current_dataset_index])

                batch = current_dataset.select(list(range(start, end)))

                self.progress_bars.update(self.tasks[current_dataset_index], completed=end)

                yield self._prepare_batch(batch, dataset=self.paths[current_dataset_index])

                datasets_sample_counter[current_dataset_index] = end
                idx += 1

                self.log_progress_prime(self.paths, datasets_sample_counter)

                if sum(datasets_sample_counter) >= self.total_samples:
                    break

    def log_progress_prime(self, paths: list[str], dataset_counters: list[int]):
        metric = {path: counter for path, counter in zip(paths, dataset_counters)}
        metric.update({"total": sum(dataset_counters)})
        self.prime_metric.log_prime(metric)
