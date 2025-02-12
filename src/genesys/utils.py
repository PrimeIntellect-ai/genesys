from datetime import datetime
import json
import os
import re
import string
import random
import base64
import time
import torch
import socket
import threading
import requests

from google.cloud import storage
from google.oauth2 import service_account
from queue import Queue
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.console import Console
from huggingface_hub import snapshot_download
from datasets import load_dataset
from pathlib import Path


class GcpBucket:
    def __init__(self, gcp_path: str, credentials_base64: str):
        # Parse GCS path (e.g., "gs://bucket-name/folder/path")
        path = gcp_path.replace("gs://", "")
        self.bucket_name = path.split("/")[0]
        self.destination_folder = "/".join(path.split("/")[1:])

        credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

        self.client = storage.Client(credentials=credentials)
        self.bucket = self.client.bucket(self.bucket_name)
        print(f"Initialized GCP bucket: {self.bucket_name}, folder: {self.destination_folder}")

        self.upload_queue = Queue()
        self._start_upload_worker()

    def _start_upload_worker(self):
        def worker():
            while True:
                file_name = self.upload_queue.get()
                if file_name is None:
                    break
                try:
                    self._push(file_name=file_name)
                except Exception as e:
                    print(f"Error uploading {file_name}: {e}")
                self.upload_queue.task_done()

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _push(self, file_name: str):
        # Create the full destination path including folder
        destination_blob_name = os.path.join(self.destination_folder, os.path.basename(file_name))
        print(f"Uploading {file_name} to gs://{self.bucket_name}/{destination_blob_name}")

        # Upload the file
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_name)
        print(f"Successfully uploaded {file_name} to GCP bucket {self.bucket_name}/{destination_blob_name}")

    def push(self, file_name: str):
        self.upload_queue.put(file_name)

    def __del__(self):
        if hasattr(self, "upload_queue"):
            self.upload_queue.put(None)
        if hasattr(self, "worker_thread"):
            self.worker_thread.join()


def save_batch_results(batch_results, results_file: "str | Path", gcp_bucket: GcpBucket | None = None):
    # Save locally first
    with open(results_file, "a") as f:
        for result in batch_results:
            json.dump(result, f)
            f.write("\n")

    # Upload to GCP if bucket is configured
    if gcp_bucket is not None:
        try:
            gcp_bucket.push(results_file)
        except Exception as e:
            print(f"Error uploading to GCP: {str(e)}")


def generate_short_id(length=8):
    """Generate a short random ID."""
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def display_config_panel(console: Console, config):
    # Create title panel
    title = Text("Genesys LLM Generator", justify="center")
    console.print(Panel(title, box=box.ROUNDED, width=70))

    # Create configuration content
    config_text = Text()
    config_text.append("\n")  # Initial spacing
    config_text.append("  Model           ", style="default")
    config_text.append(f"{config.name_model}\n", style="blue")
    config_text.append("  GPUs            ", style="default")
    config_text.append(f"{config.num_gpus}\n", style="green")
    config_text.append("  Max Tokens      ", style="default")
    config_text.append(f"{config.max_tokens:,}\n", style="yellow")
    config_text.append("  Temperature     ", style="default")
    config_text.append(f"{config.temperature}\n", style="magenta")
    config_text.append("  Top P     ", style="default")
    config_text.append(f"{config.top_p}\n", style="navy_blue")
    config_text.append("  Batch Size      ", style="default")
    config_text.append(f"{config.data.batch_size}\n", style="cyan")
    config_text.append("  Dataset         ", style="default")
    config_text.append(f"{config.data.path}\n", style="cyan")
    config_text.append("\n")  # Final spacing

    # Create configuration panel
    console.print(Panel(config_text, title="Configuration", box=box.ROUNDED, width=70))


def extract_json(text):
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If no triple backticks, try to find content between curly braces
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("No JSON-like content found in the markdown")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from the extracted content")


def get_machine_info() -> dict[str, str | int]:
    """
    gather info about the node we're running on
    """
    try:
        with open("/etc/machine-id", "r") as f:
            machine_id = f.read().strip()
    except Exception:
        machine_id = None

    try:
        num_gpus = torch.cuda.device_count()
        gpu_device_list = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    except Exception:
        num_gpus = 0
        gpu_device_list = []

    try:
        global_ipv4 = requests.get("https://icanhazip.com", timeout=5).text.strip()
    except Exception:
        global_ipv4 = None

    try:
        global_ipv6 = socket.getaddrinfo("icanhazip.com", 443, socket.AF_INET6)[0][4][0]
    except Exception:
        global_ipv6 = None

    info_dict = {
        "machine_id": machine_id,
        "num_gpus": num_gpus,
        "gpu_device_list": gpu_device_list,
        "global_ipv4": global_ipv4,
        "global_ipv6": global_ipv6,
    }

    return info_dict


def download_model(name_model: str, pre_download_retry: int):
    """Download a model from HuggingFace Hub with retries with exponential backoff.

    Args:
        name_model (str): HuggingFace model name
        pre_download_retry (int): Number of retry attempts.
    """
    console = Console()

    for i in range(pre_download_retry):
        try:
            snapshot_download(repo_id=name_model, local_files_only=False, resume_download=True)
            break
        except Exception as e:
            wait_times = [5, 30, 300]
            t = wait_times[min(i, len(wait_times) - 1)]

            console.print(f"[red]Failed to pre-download model, retrying in {t} seconds [/]")
            console.print(f"[red]Error: {e}[/]")
            time.sleep(t)

            if i == pre_download_retry - 1:
                raise e


def load_dataset_ft(path: str, retry):  # -> Dataset | List | Any | None:
    """Load a dataset from HuggingFace Hub with retries with exponential backoff.

    Args:
        path (str): HuggingFace dataset path/name
        retry (int): Number of retry attempts.
    """
    console = Console()

    for i in range(retry):
        try:
            return load_dataset(path)["train"]
        except Exception as e:
            wait_times = [2, 10, 60]
            t = wait_times[min(i, len(wait_times) - 1)]

            console.print(f"[red]Failed to pre-download model, retrying in {t} seconds [/]")
            console.print(f"[red]Error: {e}[/]")
            time.sleep(t)

            if i == retry - 1:
                raise e


console = Console()


def log(message):
    """For some reason Sglang silence real pyton logger so had to use this cheap fake logging"""
    formatted_message = f"[{datetime.now().strftime('%H:%M:%S')}]: {message}"
    console.print(formatted_message)
