import socket
import platform
import json
import os


class PrimeMetric:
    def __init__(self, disable: bool = False):
        self.disable = disable

    @classmethod
    def get_default_socket_path(cls) -> str:
        """Returns the default socket path based on the operating system."""
        default = (
            "/tmp/com.prime.miner/metrics.sock"
            if platform.system() == "Darwin"
            else "/var/run/com.prime.miner/metrics.sock"
        )
        return os.getenv("PRIME_TASK_BRIDGE_SOCKET", default=default)

    def send_message_prime(self, metric: dict, socket_path: str = None) -> bool:
        """Sends a message to the specified socket path or uses the default if none is provided."""
        socket_path = socket_path or os.getenv("PRIME_TASK_BRIDGE_SOCKET", self.get_default_socket_path())
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

    def log_prime(self, metric: dict):
        if self.disable:
            return
        if not (self.send_message_prime(metric)):
            print(f"Prime logging failed: {metric}")


# def get_system_metrics() -> dict:
#     """Returns CPU and memory usage metrics."""
#     return {
#         "cpu_percent": psutil.cpu_percent(),
#         "memory_percent": psutil.virtual_memory().percent
#     }


# def log_system_metrics():
#     """Logs system metrics to Prime."""
#     log_prime(get_system_metrics())
