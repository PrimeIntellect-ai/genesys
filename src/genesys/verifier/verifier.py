from typing import List, Union
from pydantic import BaseModel, Field
from .code_test_verifier import CodeTestsVerification, verify_code
from .math_verifier import MathGroundTruthVerification, verify_math

VerificationInfo = Union[MathGroundTruthVerification, CodeTestsVerification]

def verify(instructions: List[str], responses: List[str], verification_data: List[VerificationInfo]):
    scores = []
    for instruction, response, verification in zip(instructions, responses, verification_data):
        if verification.type == "code_tests":
            result = verify_code(response, verification.test_cases, verification.language)
        
        elif verification.type == "math_groundtruth":
            result = verify_math(response, verification.ground_truth)
        
        else:
            raise ValueError(f"Unknown verification type: {verification}")
        
        scores.append(result)
        
    return scores
