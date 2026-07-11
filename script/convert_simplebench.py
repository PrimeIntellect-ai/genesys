import argparse
import json
from pathlib import Path


def convert_eval_data(eval_data):
    for item in eval_data:
        question_id = str(item["question_id"])
        answer = item["answer"].strip().upper()
        yield {
            "problem_id": f"simplebench-{question_id}",
            "source": "simple-bench/SimpleBench",
            "task_type": "simple_bench",
            "in_source_id": question_id,
            "prompt": item["prompt"],
            "gold_standard_solution": answer,
            "verification_info": {"answer": answer},
            "metadata": {"question_id": item["question_id"]},
        }


def main():
    parser = argparse.ArgumentParser(description="Convert SimpleBench JSON to Genesys task JSONL.")
    parser.add_argument("input", type=Path, help="Path to simple_bench_public.json")
    parser.add_argument("output", type=Path, help="Path to write Genesys-compatible JSONL")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    with args.output.open("w", encoding="utf-8") as f:
        for task in convert_eval_data(payload["eval_data"]):
            f.write(json.dumps(task, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
