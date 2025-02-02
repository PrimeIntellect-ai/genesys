import json
import ast
import asyncio
from typing import List, Dict, Any, Callable
from tqdm.asyncio import tqdm
from pydantic_config import BaseConfig, parse_argv
from genesys.schemas import UnscoredResult
from genesys.verifiers.registry import VERIFIER_REGISTRY


class VerifyConfig(BaseConfig):
    file: str


async def run_sync_with_timeout(sync_func: Callable, result: UnscoredResult, timeout=None):
    """
    Runs a synchronous function in an executor with optional timeout.
    """
    loop = asyncio.get_running_loop()
    coro = loop.run_in_executor(None, sync_func, result)
    if timeout is not None:
        return await asyncio.wait_for(coro, timeout=timeout)
    return await coro


async def verify(results: List[UnscoredResult]) -> List[Any]:
    """
    Given a list of result dictionaries, dispatch each to the appropriate verifier
    (as determined by the "task_type" field) and run them concurrently using semaphores
    based on each verifier's max_parallel setting. Returns a list of verification scores.
    """
    task_types_in_use = set(item["task_type"] for item in results)

    verifier_instances = {}
    for ttype in task_types_in_use:
        verifier_instances[ttype] = VERIFIER_REGISTRY[ttype]()

    semaphores = {ttype: asyncio.Semaphore(verifier_instances[ttype].max_parallel) for ttype in task_types_in_use}

    verification_results = [None] * len(results)

    async def process_result(index: int, result: Dict):
        ttype = result["task_type"]
        verifier_obj = verifier_instances[ttype]
        async with semaphores[ttype]:
            try:
                verification_result = await run_sync_with_timeout(verifier_obj.verify, result, timeout=200)
                verification_results[index] = verification_result
            except asyncio.TimeoutError:
                print(f"Timeout verifying '{ttype}' at index {index}")
                verification_results[index] = {"score": None, "verification_result_info": {"failure_reason": "timeout"}}

    tasks = [asyncio.create_task(process_result(i, r)) for i, r in enumerate(results)]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying"):
        await task

    return verification_results


async def async_main(config: VerifyConfig):
    """
    Reads the input file (one JSON object per line), converts certain fields using ast.literal_eval,
    and then calls the verify() function to compute scores. The output is written to a new file.
    """
    to_verify = []
    with open(config.file, "r") as f:
        for line in f:
            d = json.loads(line)
            d["verification_info"] = ast.literal_eval(d["verification_info"])
            d["metadata"] = ast.literal_eval(d["metadata"])
            to_verify.append(d)

    verification_results = await verify(to_verify)

    all_results = []
    for v, d in zip(verification_results, to_verify):
        d["score"] = v["score"]
        d["verification_result_info"] = v["verification_result_info"]
        all_results.append(d)

    out_file = f"{config.file.split('.json')[0]}_verified.jsonl"
    with open(out_file, "w") as f:
        for result in all_results:
            json.dump(result, f)
            f.write("\n")


def main(config: VerifyConfig):
    asyncio.run(async_main(config))


if __name__ == "__main__":
    config = VerifyConfig(**parse_argv())
    main(config)
