from genesys.verifiers.simple_bench_verifier import SimpleBenchVerifier


def make_result(llm_response, answer="C"):
    return {
        "llm_response": llm_response,
        "verification_info": {"answer": answer},
        "gold_standard_solution": answer,
    }


def test_simple_bench_verifier_accepts_final_answer_format():
    verifier = SimpleBenchVerifier()

    result = verifier.verify(make_result("Reasoning...\nFinal Answer: C"))

    assert result["score"] == 1
    assert result["verification_result_info"]["extracted_answer"] == "C"


def test_simple_bench_verifier_accepts_parenthesized_answer():
    verifier = SimpleBenchVerifier()

    result = verifier.verify(make_result("After checking the options, final answer is (c)."))

    assert result["score"] == 1


def test_simple_bench_verifier_rejects_wrong_answer():
    verifier = SimpleBenchVerifier()

    result = verifier.verify(make_result("Final Answer: B"))

    assert result["score"] == 0
