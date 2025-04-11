"""
returns 1 if compilation and correctness pass, 0 if not. 
returns verification_result_info, a dict with details such as runtime statistics and error messages.
"""

import torch   
from typing import Dict
from genesys.schemas import Response
from genesys.verifiers.base_verifier import BaseVerifier
import modal

from genesys.local_kernel_eval import eval_kernel_against_ref, set_gpu_arch

image = modal.Image.debian_slim().pip_install(
    "pydantic",
    "torch",
    "h2",
    "grpclib",
    "datasets",
    "certifi",
    "google-cloud-storage",
    "grpcio",
    "rich",
    "huggingface_hub"
).add_local_python_source("genesys")

app = modal.App("genesys_kernelbench_modal_verifier", image=image)

@app.cls()
class EvalFunc:
    @modal.method()
    def eval_single_sample_modal(
        self,
        ref_arch_src: str,
        custom_cuda: str,
        verbose: bool,
        gpu_arch: list
    ) -> dict:
        import torch
        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")
        return eval_kernel_against_ref(
            ref_arch_src=str(ref_arch_src),    
            custom_cuda=custom_cuda,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=5,
            num_perf_trials=100,
            device=device,
            build_dir=None
        )

class KernelBenchModalVerifier(BaseVerifier):
    """
    Expected Response object fields:
      - llm_response: Candidate kernel code (string).
      - verification_info: Dict with "ref_arch_src" holding the reference kernel code.
      - metadata: Dict containing:
            "gpu": GPU type (e.g., "L40S" or "NVIDIA L40S"; the prefix is stripped),
            "gpu_arch": e.g., ["Ada"],
            "verbose": bool.
    
    verify() launches the Modal sandbox to compile and run both kernels.
    If both the candidate compiles and its outputs match the reference (within tolerance),
    the verifier returns a score of 1.0; otherwise, 0.0.
    Additional details (e.g., runtime stats and error messages) are in "verification_result_info".
    """
    max_parallel = 1
    timeout = 300   

    def verify(self, result: Response) -> Dict:
        candidate_kernel = result.llm_response
        ref_arch_src = result.verification_info.get("ref_arch_src")
        if not ref_arch_src:
            raise ValueError("Missing 'ref_arch_src' in verification_info for KernelBenchModalVerifier.")

        ref_arch_src = str(ref_arch_src)
        
        verbose = result.metadata.get("verbose", False)
        gpu_raw = result.metadata.get("gpu", "L40S")
        gpu = gpu_raw.replace("NVIDIA ", "")
        gpu_arch = result.metadata.get("gpu_arch", ["Ada"])

        with app.run():
            kernel_exec_result = EvalFunc.with_options(gpu=gpu)().eval_single_sample_modal.remote(
                ref_arch_src, candidate_kernel, verbose, gpu_arch
            )

        compiled = kernel_exec_result.get("compiled", False)
        correct = kernel_exec_result.get("correct", False)
        score = 1.0 if compiled and correct else 0.0
        return {
            "score": score,
            "verification_result_info": kernel_exec_result,
        }


#testing the pipeline with fake response, for real online RL with inference time kernel generation, 

if __name__ == "__main__":
    import json
    from pathlib import Path
    from datasets import load_dataset

    kernel_file = Path("ref_kernel.json")
    if kernel_file.exists():
        print("[TEST] Found local kernel.json; using its contents.")
        with kernel_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        candidate_kernel = data.get("kernel") or data.get("kernel_code") or data.get("result")
        if not candidate_kernel:
            raise ValueError("No kernel code found in kernel.json. Available keys: " + str(list(data.keys())))
        hardware = data.get("hardware", "L40S")
        gpu = hardware.replace("NVIDIA ", "")
        FakeResponse = type("FakeResponse", (), {})()
        FakeResponse.llm_response = candidate_kernel
        FakeResponse.verification_info = {"ref_arch_src": str(candidate_kernel)}
        FakeResponse.metadata = {
            "gpu": gpu,
            "gpu_arch": ["Ada"],
            "verbose": True,
        }
    else:
        print("[TEST] No local kernel.json; loading sample from HF dataset.")
        try:
            dataset = load_dataset("ScalingIntelligence/kernelbench-samples", split="test", streaming=True)
            sample = next(iter(dataset))
        except Exception as e:
            print("Error loading dataset:", e)
            exit(1)
        candidate_kernel = sample.get("kernel") or sample.get("kernel_code") or sample.get("result")
        if not candidate_kernel:
            raise ValueError("No kernel code found in sample. Available keys: " + str(list(sample.keys())))
        hardware = sample.get("hardware", "L40S")
        gpu = hardware.replace("NVIDIA ", "")
        FakeResponse = type("FakeResponse", (), {})()
        FakeResponse.llm_response = candidate_kernel
        FakeResponse.verification_info = {"ref_arch_src": str(candidate_kernel)}
        FakeResponse.metadata = {
            "gpu": gpu,
            "gpu_arch": ["Ada"],
            "verbose": True,
        }

    verifier = KernelBenchModalVerifier()
    result = verifier.verify(FakeResponse)
    print("Final Evaluation Result (Testing Block):")
    print(json.dumps(result, indent=2))
