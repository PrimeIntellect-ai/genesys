import json
import os
import string
import random
from google.cloud import storage
from queue import Queue
import threading


class GcpBucket:
    def __init__(self, gcp_path: str):
        # Parse GCS path (e.g., "gs://bucket-name/folder/path")
        path = gcp_path.replace("gs://", "")
        self.bucket_name = path.split("/")[0]
        self.destination_folder = "/".join(path.split("/")[1:])

        # Initialize client
        self.client = storage.Client()
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


def save_batch_results(batch_results, results_file, gcp_bucket: GcpBucket | None = None):
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
