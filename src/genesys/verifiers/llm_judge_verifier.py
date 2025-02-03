import os
from typing import Tuple, Dict
from openai import OpenAI
from genesys.utils import extract_json
from genesys.verifiers.base_verifier import BaseVerifier

# Global constant for the judge prompt.
JUDGE_PROMPT = """
Your job is to judge the output of a large language model trained to reason about hard problems such as mathematics, science, and coding. Below you will see a problem given to the model, a gold standard response provided by a human, and the model's provided response. 
Your job is to grade the model response's correctness by how close it is to the content of the gold standard response on a scale of 1-100. The rating should NOT be about the similarity of the style, but about whether the llm response is given the same factual statements as the gold standard response.
The grading should be quite strict, you shouldn't just give every response a score of 70 or better. Only give high scores if the model actually matches the gold standard response.
{additional_judging_instructions}
Here is the problem:
{problem}

---

Here is the gold standard response:
{gold_standard_response}

---

Here is the response the model provided:
{llm_response}

---

Think about your score and then return a json with a first field 'explanation' explaining your score and a second field 'score' indicating how close the content of the LLM response is to the human response on a scale of 1-100.
"""


class LlmJudgeVerifier(BaseVerifier):
    max_parallel = 30  # For concurrency control if needed.
    timeout = 120

    def verify(self, result: Dict) -> Tuple[float, str]:
        """
        Uses an LLM judge to compare the model's response with a gold standard response.
        Expects a result dictionary containing:
          - "prompt": the problem statement.
          - "llm_response": the model's output.
          - "gold_standard_solution": the gold standard output.
          - "verification_info": a dict with a "judging_instructions" field.

        Returns:
            A tuple (score, explanation), where the score is normalized to the range 0-1.
        """
        problem = result["prompt"]
        llm_response = result["llm_response"]
        gold_standard_response = result["gold_standard_solution"]
        judging_instructions = result["verification_info"]["judging_instructions"]

        model_name = "gpt-4o"
        provider = "openai"

        client = self._init_client(provider)

        prompt = JUDGE_PROMPT.format(
            problem=problem,
            gold_standard_response=gold_standard_response,
            llm_response=llm_response,
            additional_judging_instructions=self._build_additional_judging_instructions(judging_instructions),
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        gen_json = extract_json(response.choices[0].message.content)

        return dict(
            score=gen_json["score"] / 100, verification_result_info=dict(judge_explanation=gen_json["explanation"])
        )

    def _init_client(self, provider: str) -> OpenAI:
        """
        Initialize and return an OpenAI client based on the given provider.
        """
        if provider == "together":
            return OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
        if provider == "openai":
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        raise ValueError(f"Unsupported provider: {provider}")

    def _build_additional_judging_instructions(self, judging_instructions: str) -> str:
        """
        If additional judging instructions are provided, format them for inclusion in the prompt.
        """
        if judging_instructions == "":
            return ""
        return f"\nHere are additional judging instructions:\n{judging_instructions}\n\n"
