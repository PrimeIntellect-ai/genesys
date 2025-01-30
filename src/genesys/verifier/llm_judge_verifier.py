from openai import OpenAI
import os
import re
import json

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


def init_client(provider):
    if provider == "together":
        return OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
    if provider == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def build_additional_judging_instructions(judging_instructions):
    if judging_instructions == "":
        return ""

    return f"\nHere are additional judging instructions:\n{judging_instructions}\n\n"


def extract_json(text):
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If no triple backticks, try to find content between curly braces
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("No JSON-like content found in the markdown")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from the extracted content")


def verify_with_llm_judge_and_groundtruth(
    problem, llm_response, gold_standard_response, judging_instructions, model_name, provider
):
    client = init_client(provider)

    prompt = JUDGE_PROMPT.format(
        problem=problem,
        gold_standard_response=gold_standard_response,
        llm_response=llm_response,
        additional_judging_instructions=build_additional_judging_instructions(judging_instructions),
    )

    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )

    gen_json = extract_json(response.choices[0].message.content)

    return gen_json["score"] / 100, gen_json["explanation"]
