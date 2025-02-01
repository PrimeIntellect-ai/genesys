import asyncio
from typing import List, Dict
from tqdm.asyncio import tqdm
from genesys.verifier.code_test_verifier import verify_code, init_containers, close_containers
from genesys.verifier.math_verifier import verify_math
from genesys.verifier.llm_judge_verifier import verify_with_llm_judge_and_groundtruth
from genesys.verifier.code_output_prediction_verifier import verify_code_understanding


async def async_verify_code(response: str, test_cases, language):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, verify_code, response, test_cases, language)


async def async_verify_llm_judge_comparison(problem, response, gold_standard_response, judging_instructions):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        verify_with_llm_judge_and_groundtruth,
        problem,
        response,
        gold_standard_response,
        judging_instructions,
        "gpt-4o",
        "openai",
    )


async def async_verify_math(model_output, ground_truth_answer, timeout_seconds=5):
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, verify_math, model_output, ground_truth_answer), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        print(f"Timed out verifying math for ground truth: {ground_truth_answer}")
        return False


async def async_verify_code_understanding(model_output, ground_truth_answer):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, verify_code_understanding, model_output, ground_truth_answer)


async def verify(
    results,
    max_parallel: Dict[str, int] = {
        "verifiable_code": 5,
        "verifiable_math": 10,
        "llm_judgeable_groundtruth_similarity": 30,
        "verifiable_code_understanding": 30,
    },
    math_timeout: int = 5,
) -> List[float]:
    task_types = [r["task_type"] for r in results]
    has_code_tests = any(task_type == "verifiable_code" for task_type in task_types)

    if has_code_tests:
        init_containers()

    semaphores = {
        "verifiable_code": asyncio.Semaphore(max_parallel.get("verifiable_code", 5)),
        "verifiable_math": asyncio.Semaphore(max_parallel.get("verifiable_math", 10)),
        "llm_judgeable_groundtruth_similarity": asyncio.Semaphore(
            max_parallel.get("llm_judgeable_groundtruth_similarity", 30)
        ),
        "verifiable_code_understanding": asyncio.Semaphore(max_parallel.get("verifiable_code_understanding", 30)),
    }

    scores = [None] * len(results)

    async def process_result(index: int, result):
        task_type = result["task_type"]
        async with semaphores[task_type]:
            if task_type == "verifiable_code":
                score = await async_verify_code(
                    result["llm_response"],
                    result["verification_info"]["test_cases"],
                    result["verification_info"]["language"],
                )
            elif task_type == "verifiable_math":
                score = await async_verify_math(
                    result["llm_response"], result["verification_info"]["ground_truth"], timeout_seconds=math_timeout
                )
            elif task_type == "llm_judgeable_groundtruth_similarity":
                score, _ = await async_verify_llm_judge_comparison(
                    result["prompt"],
                    result["llm_response"],
                    result["gold_standard_solution"],
                    result["verification_info"]["judging_instructions"],
                )
            elif task_type == "verifiable_code_understanding":
                score = await async_verify_code_understanding(
                    result["llm_response"], result["verification_info"]["ground_truth"], timeout_seconds=math_timeout
                )

            else:
                raise ValueError(f"Unknown verification type: {task_type}")
            scores[index] = score

    try:
        tasks = [asyncio.create_task(process_result(index, result)) for index, result in enumerate(results)]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying"):
            await task

    finally:
        if has_code_tests:
            close_containers()

    return scores
