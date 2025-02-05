import torch
import sys


def get_model_name(device_name: str) -> str:
    """find out which model we can run"""
    if "A100" in device_name:
        return "llama70b"
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return "llama70b"
        elif "PCIe" in device_name:
            return "llama70b"
        else:  # for H100 SXM and other variants
            return "llama70b"
    elif "H200" in device_name:
            return "deepseek_r1"
    else:  # for other GPU types, assume A100
        return "llama70b"
    
if __name__ == "__main__":
    try:
        device_info_torch = torch.cuda.get_device_name(torch.device("cuda:0"))
        run_model_name = get_model_name(device_info_torch)
        print(run_model_name)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)