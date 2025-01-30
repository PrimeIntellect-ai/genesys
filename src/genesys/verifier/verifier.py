import asyncio
import json
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
from genesys.verifier.code_test_verifier import verify_code, init_containers, close_containers
from genesys.verifier.math_verifier import verify_math
from genesys.verifier.llm_judge_verifier import verify_with_llm_judge_and_groundtruth

async def async_verify_code(response: str, test_cases, language):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, verify_code, response, test_cases, language)

async def async_verify_llm_judge_comparison(problem, response, gold_standard_response, judging_instructions):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, verify_with_llm_judge_and_groundtruth, problem, response, gold_standard_response, judging_instructions, "gpt-4o", "openai")

async def async_verify_math(model_output, ground_truth_answer, timeout_seconds=5):
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, verify_math, model_output, ground_truth_answer),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        print(f"Timed out verifying math for ground truth: {ground_truth_answer}")
        return False

async def verify(
    results,
    max_parallel: Dict[str, int] = {"verifiable_code": 5, "verifiable_math": 10, "llm_judgeable_groundtruth_similarity": 30},
    math_timeout: int = 5
) -> List[float]:

    task_types = [r["task_type"] for r in results]
    has_code_tests = any(task_type == "verifiable_code" for task_type in task_types)

    if has_code_tests:
        init_containers()

    semaphores = {
        "verifiable_code": asyncio.Semaphore(max_parallel.get("verifiable_code", 5)),
        "verifiable_math": asyncio.Semaphore(max_parallel.get("verifiable_math", 10)),
        "llm_judgeable_groundtruth_similarity": asyncio.Semaphore(max_parallel.get("llm_judgeable_groundtruth_similarity", 30)),
    }

    scores = [None] * len(results)

    async def process_result(index: int, result):
        task_type = result["task_type"]
        async with semaphores[task_type]:
            if task_type == "verifiable_code":
                score = await async_verify_code(
                    result["llm_response"],
                    result["verification_info"]["test_cases"],
                    result["verification_info"]["language"]
                )
            elif task_type == "verifiable_math":
                score = await async_verify_math(
                    result["llm_response"],
                    result["verification_info"]["ground_truth"],
                    timeout_seconds=math_timeout
                )
            elif task_type == "llm_judgeable_groundtruth_similarity":
                score, _ = await async_verify_llm_judge_comparison(
                    result["prompt"],
                    result["llm_response"],
                    result["gold_standard_solution"],
                    result["verification_info"]["judging_instructions"]
                )

            else:
                raise ValueError(f"Unknown verification type: {task_type}")
            scores[index] = score

    try:
        tasks = [
            asyncio.create_task(process_result(index, result))
            for index, result in enumerate(results)
        ]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying"):
            await task

    finally:
        if has_code_tests:
            close_containers()

    return scores

if __name__ == "__main__":

    sample_results = [
        {
            "task_type": "verifiable_math",
            "llm_response": "The answer is \\boxed{42}.",
            "verification_info": {
                "ground_truth": "42"
            }
        },
        {
            "task_type": "verifiable_math",
            "llm_response": "The answer is \\boxed{32}.",
            "verification_info": {
                "ground_truth": "42"
            }
        },
        {
            "task_type": "llm_judgeable_groundtruth_similarity",
            "prompt": "What is the function in python with which i can write to stdout",
            "llm_response": "The function is write()",
            "gold_standard_solution": "The function is print()",
            "verification_info": {
                "judging_instructions": ""
            }
        },
        {
            "task_type": "llm_judgeable_groundtruth_similarity",
            "prompt": "What is the function in python with which i can write to stdout",
            "llm_response": "The function is print()",
            "gold_standard_solution": "The function is print()",
            "verification_info": {
                "judging_instructions": ""
            }
        },
        {
            "task_type": "llm_judgeable_groundtruth_similarity",
            "prompt": "What is the function in python with which i can write to stdout",
            "llm_response": "The function is print() or write(), not sure",
            "gold_standard_solution": "The function is print()",
            "verification_info": {
                "judging_instructions": ""
            }
        },
    ]

    verified_scores = asyncio.run(
        verify(
            results=sample_results,
            max_parallel={"verifiable_code": 3, "verifiable_math": 5},
            math_timeout=5
        )
    )

    print("Verification Scores:", verified_scores)
