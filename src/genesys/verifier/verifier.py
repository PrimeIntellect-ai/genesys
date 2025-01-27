from typing import List, Union
from genesys.verifier.code_test_verifier import CodeTestsVerification, verify_code, init_containers, close_containers
from genesys.verifier.math_verifier import MathGroundTruthVerification, verify_math

VerificationInfo = Union[MathGroundTruthVerification, CodeTestsVerification]


def verify(responses: List[str], verification_data: List[VerificationInfo], task_types: List[str]):
    has_code_tests = any(verification.type == "code_tests" for verification in verification_data)
    if has_code_tests:
        init_containers()

    scores = []
    for response, verification, task in zip(responses, verification_data, task_types):
        if task == "verifiable_code":
            result = verify_code(response, verification.test_cases, verification.language)

        elif task == "verifiable_math":
            result = verify_math(response, verification.ground_truth)

        else:
            raise ValueError(f"Unknown verification type: {verification}")

        scores.append(result)

    if has_code_tests:
        close_containers()

    return scores
