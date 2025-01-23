import json


def repeat_elements(lst, n):
    return [item for item in lst for _ in range(n)]


def save_batch_results(batch_results, results_file):
    with open(results_file, "a") as f:
        for result in batch_results:
            json.dump(result, f)
            f.write("\n")
