import json
import os
import panda as pd
import string
import random
import base64
from google.cloud import storage
from google.oauth2 import service_account
from queue import Queue
import threading

from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.console import Console


import socket
import platform


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



def save_batch_results(batch_results, results_file, gcp_bucket=None):
    """Save results to a Parquet file and optionally upload to GCP."""
    df = pd.DataFrame(batch_results)
    df.to_parquet(results_file, engine='pyarrow', compression='zstd')
    
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


def get_default_socket_path() -> str:
    """Returns the default socket path based on the operating system."""
    default = (
        "/tmp/com.prime.miner/metrics.sock"
        if platform.system() == "Darwin"
        else "/var/run/com.prime.miner/metrics.sock"
    )
    return os.getenv("PRIME_TASK_BRIDGE_SOCKET", default=default)


def send_message_prime(metric: dict, socket_path: str = None) -> bool:
    """Sends a message to the specified socket path or uses the default if none is provided."""
    socket_path = socket_path or os.getenv("PRIME_TASK_BRIDGE_SOCKET", get_default_socket_path())
    # print("Sending message to socket: ", socket_path)

    task_id = os.getenv("PRIME_TASK_ID", None)
    if task_id is None:
        print("No task ID found, skipping logging to Prime")
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(socket_path)

            for key, value in metric.items():
                message = {"label": key, "value": value, "task_id": task_id}
                sock.sendall(json.dumps(message).encode())
        return True
    except Exception:
        return False


def log_prime(metric: dict):
    if not (send_message_prime(metric)):
        print(f"Prime logging failed: {metric}")
