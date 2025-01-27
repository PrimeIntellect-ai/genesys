import json
import os
from google.cloud import storage
import subprocess

def repeat_elements(lst, n):
    return [item for item in lst for _ in range(n)]


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

    def push(self, file_name: str):
        # Create the full destination path including folder
        destination_blob_name = os.path.join(self.destination_folder, os.path.basename(file_name))
        print(f"Uploading {file_name} to gs://{self.bucket_name}/{destination_blob_name}")

        # Upload the file
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_name)


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
            print(f"Successfully uploaded {results_file} to GCP bucket")
        except Exception as e:
            print(f"Error uploading to GCP: {str(e)}")


def get_gpu_memory():
    """
    Retrieve the total GPU memory available across all GPUs in the system in megabytes

    This function uses the `nvidia-smi` command to query the total memory
    available for each GPU installed in the system. The memory values are 
    summed up to provide the total memory in megabytes (MB).

    Returns:
        int: The total GPU memory in MB if successful.
        None: If an error occurs (e.g., `nvidia-smi` not found or other exceptions).

    Example:
        >>> get_gpu_memory()
        24576  # Total GPU memory in MB across all GPUs
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        memory_list = [int(x) for x in output.decode().strip().split('\n')]
        total_memory_mb = sum(memory_list)
        return total_memory_mb
    except Exception as e:
        print(f"Error: {e}")
        return None
