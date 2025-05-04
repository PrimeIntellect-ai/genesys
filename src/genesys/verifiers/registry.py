from genesys.verifiers.code_test_verifier import CodeVerifier
from genesys.verifiers.math_verifier import MathVerifier
from genesys.verifiers.llm_judge_verifier import LlmJudgeVerifier
from genesys.verifiers.code_output_prediction_verifier import CodeUnderstandingVerifier
from genesys.verifiers.swe_fixer_verifier import SweFixerVerifier

VERIFIER_REGISTRY = {
    "verifiable_code": CodeVerifier,
    "verifiable_math": MathVerifier,
    "llm_judgeable_groundtruth_similarity": LlmJudgeVerifier,
    "code_output_prediction": CodeUnderstandingVerifier,
    "swe_fixer": SweFixerVerifier,
}
