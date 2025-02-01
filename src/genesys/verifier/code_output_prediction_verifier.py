from genesys.utils import extract_json


# a very small verifier
def verify_code_understanding(llm_response: str, ground_truth: str):
    output_prediction = extract_json(llm_response)["output"]

    return int(output_prediction == ground_truth)
