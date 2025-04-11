import os
import sys
import torch
import torch.nn as nn
import torch.utils.cpp_extension
import subprocess
import random
import json
import time
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Optional, Dict, Any


def set_gpu_arch(gpu_arch):
    """
    If you need architecture-specific compilation flags, you can set them here.
    For example:
        if "Ada" in gpu_arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    Right now this is a no-op by default.
    """
    pass


def _cleanup_cuda_extensions():
    """
    Optional: remove compiled CUDA extensions from local cache if needed.
    Currently not used by default.
    """
    import shutil
    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)

def _graceful_cleanup(context: dict, device: torch.device):
    """
    Clean up references, GPU cache, etc.
    """
    del context
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)


def _compile_and_exec_python_code(py_code: str, context: dict):
    """
    Compile and exec Python code string in a context dictionary.
    """
    # first compile to check for syntax
    compile(py_code, "<string>", "exec")
    exec(py_code, context)

def _load_custom_model(
    model_custom_src: str,
    context: dict,
    build_directory: Optional[str] = None,
):
    """
    Load a custom nn.Module from Python code that includes inline CUDA extension calls.
    If build_directory is specified, we set the TORCH_EXTENSIONS_DIR env var to compile there.
    """
    if build_directory:
        preamble = (
            "import os\n"
            f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        )
        model_custom_src = preamble + model_custom_src

    _compile_and_exec_python_code(model_custom_src, context)
    ModelNew = context.get("ModelNew", None)
    if ModelNew is None:
        raise RuntimeError("No class 'ModelNew' found in the candidate code.")
    return ModelNew

def _load_original_model_and_inputs(model_original_src: str, context: dict):
    """
    Load a reference architecture from Python code that defines:
       Model -> a torch.nn.Module (the reference)
       get_init_inputs() -> returns a list of initialization inputs
       get_inputs() -> returns a list of test inputs
    """
    _compile_and_exec_python_code(model_original_src, context)
    Model = context.get("Model", None)
    get_init_inputs_fn = context.get("get_init_inputs", None)
    get_inputs_fn = context.get("get_inputs", None)
    if any(x is None for x in [Model, get_init_inputs_fn, get_inputs_fn]):
        raise RuntimeError("Reference code must define Model, get_init_inputs, get_inputs.")
    return (Model, get_init_inputs_fn, get_inputs_fn)

def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Measures execution time (ms) of kernel_fn(*args) using CUDA events.
    """
    if device is None:
        device = torch.device("cuda")

    # Warm up
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    if verbose:
        print(f"[Timing] Warmup={num_warmup}, Trials={num_trials}, device={device}")

    times = []
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        kernel_fn(*args)
        end_event.record()
        torch.cuda.synchronize(device=device)
        elapsed_ms = start_event.elapsed_time(end_event)  # ms
        times.append(elapsed_ms)
        if verbose:
            print(f"Trial {trial+1}: {elapsed_ms:.3f} ms")

    return times

def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """
    Return {mean, std, min, max, num_trials, hardware, device} from list of ms times
    """
    if not elapsed_times:
        return {}
    arr = np.array(elapsed_times)
    stats = {
        "mean": float(f"{arr.mean():.3g}"),
        "std": float(f"{arr.std():.3g}"),
        "min": float(f"{arr.min():.3g}"),
        "max": float(f"{arr.max():.3g}"),
        "num_trials": len(elapsed_times),
    }
    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)
    return stats

def run_correctness_trials(
    ref_model: nn.Module,
    candidate_model: nn.Module,
    get_inputs_fn: callable,
    device: torch.device,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    metadata: dict = None,
):
    """
    Run multiple correctness trials with different random seeds, using get_inputs_fn to create test inputs.
    Compare candidate output with reference output using torch.allclose with given tolerances.
    """
    if metadata is None:
        metadata = {}
    pass_count = 0
    # deterministically generate seeds for correctness trials
    _set_seed(seed)
    seeds = [torch.randint(0, 2**32-1, (1,)).item() for _ in range(num_correct_trials)]

    with torch.no_grad():
        for trial_idx in range(num_correct_trials):
            trial_seed = seeds[trial_idx]
            if verbose:
                print(f"[Correctness] Trial {trial_idx+1}, seed={trial_seed}")
            _set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            # Run reference
            ref_out = ref_model(*inputs)
            torch.cuda.synchronize(device=device)

            # Run candidate
            cand_out = candidate_model(*inputs)
            torch.cuda.synchronize(device=device)

            # Compare shapes
            if ref_out.shape != cand_out.shape:
                msg = f"Output shape mismatch: ref={ref_out.shape}, cand={cand_out.shape}"
                metadata["correctness_issue"] = msg
                if verbose:
                    print(f"[FAIL] trial {trial_idx+1}: {msg}")
                return False  # immediate fail

            # Compare values
            if not torch.allclose(ref_out, cand_out, atol=atol, rtol=rtol):
                # record diffs
                diff = torch.abs(ref_out - cand_out)
                max_diff = diff.max().item()
                avg_diff = diff.mean().item()
                metadata.setdefault("max_difference", []).append(f"{max_diff:.4f}")
                metadata.setdefault("avg_difference", []).append(f"{avg_diff:.4f}")
                metadata["correctness_issue"] = "output mismatch"
                if verbose:
                    print(f"[FAIL] trial {trial_idx+1}: output mismatch (max diff {max_diff:.4f})")
            else:
                pass_count += 1
                if verbose:
                    print(f"[PASS] trial {trial_idx+1}")

    metadata["correctness_trials"] = f"{pass_count} / {num_correct_trials}"
    return pass_count == num_correct_trials

def eval_kernel_against_ref(
    ref_arch_src: str,
    custom_cuda: str,
    verbose: bool = False,
    measure_performance: bool = False,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    device: torch.device = None,
    build_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate kernel code (custom_cuda) against a reference architecture (ref_arch_src).
    Returns a dict:
      {
        "compiled": bool,
        "correct": bool,
        "avg_runtime": float or None,
        "runtime_stats": dict,
        "error": str
      }

    Steps:
      1) Compile & load reference code -> yields Model, get_init_inputs, get_inputs
      2) Compile & load candidate code -> yields ModelNew
      3) Check correctness over multiple trials
      4) Measure runtime if measure_performance == True
    """
    if not torch.cuda.is_available():
        return {
            "compiled": False,
            "correct": False,
            "avg_runtime": None,
            "runtime_stats": {},
            "error": "CUDA not available"
        }
    if device is None:
        device = torch.device("cuda")

    metadata = {}
    torch.cuda.set_device(device)

    context = {}
    try:
        if verbose:
            print("[Eval] Compiling reference model code.")
        ref_model_class, get_init_inputs, get_inputs_fn = _load_original_model_and_inputs(ref_arch_src, context)

        _set_seed(42)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        ref_model = ref_model_class(*init_inputs).cuda(device=device)
        torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] Reference model loaded/compiled successfully.")
    except Exception as e:
        msg = f"Reference compilation or loading failed: {str(e)}"
        if verbose:
            print("[Eval Error] " + msg)
        _graceful_cleanup(context, device)
        return {
            "compiled": False,
            "correct": False,
            "avg_runtime": None,
            "runtime_stats": {},
            "error": msg,
        }

    try:
        if verbose:
            print("[Eval] Compiling candidate kernel code.")
        candidate_model_class = _load_custom_model(custom_cuda, context, build_dir)
        _set_seed(42)
        cand_init_inputs = get_init_inputs()
        cand_init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in cand_init_inputs
        ]
        candidate_model = candidate_model_class(*cand_init_inputs).cuda(device=device)
        torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] Candidate kernel code compiled and loaded successfully.")
    except Exception as e:
        msg = f"Candidate compilation or loading failed: {str(e)}"
        if verbose:
            print("[Eval Error] " + msg)
        _graceful_cleanup(context, device)
        return {
            "compiled": False,
            "correct": False,
            "avg_runtime": None,
            "runtime_stats": {},
            "error": msg,
        }

    correct = False
    try:
        correct = run_correctness_trials(
            ref_model=ref_model,
            candidate_model=candidate_model,
            get_inputs_fn=get_inputs_fn,
            device=device,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=42,
            atol=1e-2,
            rtol=1e-2,
            metadata=metadata,
        )
    except Exception as e:
        msg = f"Error during correctness trials: {str(e)}"
        if verbose:
            print("[Eval Error] " + msg)
        _graceful_cleanup(context, device)
        return {
            "compiled": True,
            "correct": False,
            "avg_runtime": None,
            "runtime_stats": {},
            "error": msg,
        }

    avg_runtime = None
    runtime_stats = {}
    if correct and measure_performance:
        if verbose:
            print("[Eval] Measuring performance (candidate).")
        try:
            _set_seed(42)
            perf_inputs = get_inputs_fn()
            perf_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in perf_inputs
            ]
            def cand_fn(*args):
                return candidate_model(*args)

            times = time_execution_with_cuda_event(
                cand_fn,
                *perf_inputs,
                num_warmup=3,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
            )
            runtime_stats = get_timing_stats(times, device=device)
            avg_runtime = runtime_stats.get("mean", None)
        except Exception as e:
            msg = f"Error measuring performance: {str(e)}"
            metadata["error_during_performance"] = msg
            if verbose:
                print("[Eval Perf Error] " + msg)

    _graceful_cleanup(context, device)

    return {
        "compiled": True,
        "correct": correct,
        "avg_runtime": avg_runtime,
        "runtime_stats": runtime_stats,
        "error": "",
        **metadata  # include e.g. shape mismatch info or differences
    }
