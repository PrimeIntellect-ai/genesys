import argparse
import itertools
import sglang as sgl
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from intellect.utils import repeat_elements, save_batch_results, verify_math_sample

SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with: \n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem. If the question is a multiple choice question, [answer] should be the letter indicating your correct response (e.g. \\text{A} or \\text{B})."


def main(args):
    llm = sgl.Engine(model_path=args.model, tp_size=args.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    math_dataset = load_dataset("PrimeIntellect/NuminaMath-groundtruth")["train"]
    math_dataset = math_dataset.add_column("problem_id", range(len(math_dataset)))

    sampling_params = dict(temperature=args.temperature, max_new_tokens=8192, stop=["<|eot_id|>"])

    open(args.out_file_name, "w").close()

    max_samples = args.max_samples if args.max_samples is not None else len(math_dataset)

    for i in tqdm(range(0, min(max_samples, len(math_dataset)), args.batch_size), desc="Generating data"):
        batch = math_dataset[i : min(i + args.batch_size, len(math_dataset))]
        batch_ids = list(
            itertools.chain.from_iterable([[idx] * args.num_responses_per_question for idx in batch["problem_id"]])
        )
        batch_ground_truths = list(
            itertools.chain.from_iterable([[gt] * args.num_responses_per_question for gt in batch["ground_truth"]])
        )

        batch_messages = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
            for problem in batch["problem"]
        ]
        batch_messages = repeat_elements(batch_messages, args.num_responses_per_question)
        batch_inputs = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        batch_output = llm.generate(batch_inputs, sampling_params)

        all_results = []
        for j, out in enumerate(batch_output):
            result = dict()
            result["prompt"] = batch_messages[j][1]["content"]
            result["response"] = out["text"]
            result["problem_id"] = int(batch_ids[j])
            result["ground_truth"] = batch_ground_truths[j]
            result["correct"] = verify_math_sample(out["text"], batch_ground_truths[j])

            all_results.append(result)

        save_batch_results(all_results, args.out_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process math problems in parallel")
    parser.add_argument("--model", type=str, default="Qwen/QwQ-32B-Preview")
    parser.add_argument("--out_file_name", type=str, default="out.jsonl")
    parser.add_argument("--num_responses_per_question", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    main(args)
