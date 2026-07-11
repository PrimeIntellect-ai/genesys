import re
from typing import Dict

from genesys.verifiers.base_verifier import BaseVerifier


FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*(?:is|:|-)?\s*\(?\s*([A-F])\s*\)?", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:|-)?\s*\(?\s*([A-F])\s*\)?", re.IGNORECASE),
]


class SimpleBenchVerifier(BaseVerifier):
    max_parallel = 30
    timeout = 5

    def verify(self, result: Dict):
        expected = self._expected_answer(result)
        extracted = self._extract_answer(result["llm_response"])

        return dict(
            score=int(extracted == expected),
            verification_result_info={
                "extracted_answer": extracted,
                "expected_answer": expected,
            },
        )

    def _expected_answer(self, result: Dict) -> str:
        verification_info = result.get("verification_info", {})
        answer = (
            verification_info.get("answer")
            or verification_info.get("ground_truth")
            or result.get("gold_standard_solution")
        )
        if answer is None:
            raise ValueError("SimpleBench responses require an expected A-F answer")

        answer = str(answer).strip().upper()
        if not re.fullmatch(r"[A-F]", answer):
            raise ValueError(f"Invalid SimpleBench expected answer: {answer!r}")
        return answer

    def _extract_answer(self, response: str) -> str | None:
        for pattern in FINAL_ANSWER_PATTERNS:
            match = pattern.search(response)
            if match:
                return match.group(1).upper()

        candidates = re.findall(r"\b([A-F])\b", response.upper())
        if candidates:
            return candidates[-1]
        return None
