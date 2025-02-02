from genesys.verifiers.code_test_verifier import CodeVerifier
from genesys.verifiers.math_verifier import MathVerifier
from genesys.verifiers.llm_judge_verifier import LlmJudgeVerifier
from genesys.verifiers.code_output_prediction_verifier import CodeUnderstandingVerifier

VERIFIER_REGISTRY = {
    "verifiable_code": CodeVerifier,
    "verifiable_math": MathVerifier,
    "llm_judgeable_groundtruth_similarity": LlmJudgeVerifier,
    "verifiable_code_understanding": CodeUnderstandingVerifier,
}
