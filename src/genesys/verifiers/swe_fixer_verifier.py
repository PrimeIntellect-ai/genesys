### adapted from https://github.com/InternLM/SWE-Fixer/blob/main/evaluation/code_edit.py
import json
import re
import ast
import argparse
import cydifflib

from genesys.schemas import Response
from genesys.verifiers.base_verifier import BaseVerifier


LINE_NUMBER_REGEX = re.compile(r"^\d+\s", re.MULTILINE)


def parse_json_codeblock_from_model_output(markdown_str):
    # Get everything after </think>, if it exists
    match = re.search(r"</think>(.*?)$", markdown_str, re.DOTALL)
    answer_str = match.group(1).strip() if match else markdown_str.strip()
    # Extract everything between ```json and ``` markers
    match = re.search(r"```json\s*(.*?)\s*```", answer_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return answer_str.strip()


def remove_line_numbers(content):
    return LINE_NUMBER_REGEX.sub("", content)


def remove_empty_lines(code):
    lines = code.splitlines()
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)


def check_syntax(code):
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


class SweFixerVerifier(BaseVerifier):
    """
    Verifier for the SWE-Fixer dataset.
    https://github.com/InternLM/SWE-Fixer
    """

    def apply_patches(self, files_to_modify, patches):
        """
        Apply a list of code-edit patches to an iterable of files and return the
        fully-patched workspace.

        Args:
            files_to_modify (list[dict]): items from verification_info["input"]["files to be modified"]
            patches (list[dict]): items structured like verification_info["output"]["edited code"]
                                or the model's JSON output.

        Returns
        -------
        dict[str, str]
            file-path -> patched file content
        """
        workspace = {f["file"]: remove_line_numbers(f["file content"]) for f in files_to_modify}
        failed_file_paths = []

        for patch in patches:
            file_path = patch["file"]
            snippet_old = remove_line_numbers(patch["code snippet to be modified"]).strip()
            snippet_new = patch["edited code snippet"].strip()

            current = workspace.get(file_path, "")
            if snippet_old:
                if snippet_old not in current:
                    # Model failed to localize the code snippet to be modified
                    print("Model failed to localize the code snippet to be modified")
                    failed_file_paths.append(file_path)
                    continue
                current = current.replace(snippet_old, snippet_new)
            elif current == "":  # brand-new file
                current = snippet_new
            workspace[file_path] = current

        # Set the file content to None for files that failed to be patched
        workspace = {k: None if k in failed_file_paths else v for k, v in workspace.items()}
        return workspace

    def get_diff(self, before, after):
        diff = cydifflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="")
        lines = list(diff)[2:]  # Keep relevant parts of the diff
        return "\n".join(lines)

    def score_patching(self, verification_info, json_output):
        """
        Score how well the model's patches match the expected patches.

        Args:
            verification_info (dict): Contains the original files to modify and golden patches
                                    in verification_info["input"]["files to be modified"] and
                                    verification_info["output"]["edited code"] respectively
            json_output (str): The model's output as a JSON string containing patches to apply

        Returns:
            dict: Contains:
                - score (float): 1.0 if patches match exactly, 0.0 if syntax error or localization failure
                - verification_result_info (dict): Additional info about verification failures
        """
        try:
            model_patches = json.loads(json_output)
        except Exception as e:
            return dict(score=0.0, verification_result_info={"failure_reason": f"Error in parsing JSON output: {e}"})

        try:
            original_files = verification_info["input"]["files to be modified"]
            golden_patches = verification_info["output"]["edited code"]

            original_workspace = self.apply_patches(original_files, [])
            golden_workspace = self.apply_patches(original_files, golden_patches)
            predicted_workspace = self.apply_patches(original_files, model_patches)
            # predicted_workspace = self.apply_patches(original_files, golden_patches)

            scores = []
            for file_path in golden_workspace:
                if predicted_workspace[file_path] is None:
                    scores.append(0.0)  # model failed to localize edit location
                    continue
                golden_file_content = golden_workspace[file_path]
                predicted_file_content = predicted_workspace.get(file_path, "")

                if predicted_file_content == golden_file_content:
                    scores.append(1.0)
                    continue

                syntax_ok = check_syntax(predicted_file_content)
                # syntax_ok_golden = check_syntax(golden_file_content)
                if not syntax_ok:
                    return dict(score=0.0, verification_result_info={"failure_reason": "Syntax error"})
                # if not syntax_ok_golden:
                #     return dict(score=0.0, verification_result_info={
                #         "failure_reason": "Syntax error in golden patch"
                #     })

                golden_diff = self.get_diff(before=original_workspace[file_path], after=golden_file_content)
                model_diff = self.get_diff(before=original_workspace[file_path], after=predicted_file_content)
                # print("Line counts expected:")
                # print(len(expected_file_content.splitlines()))
                # print(len(golden_diff.splitlines()))
                # print("Delta: ", len(expected_file_content.splitlines()) - len(golden_diff.splitlines()))
                # print("Line counts predicted:")
                # print(len(predicted_file_content.splitlines()))
                # print(len(model_diff.splitlines()))
                # print("Delta: ", len(predicted_file_content.splitlines()) - len(model_diff.splitlines()))
                # print("MODEL DIFF:\n", model_diff)
                # print("GOLDEN DIFF:\n", golden_diff)

                score = cydifflib.SequenceMatcher(
                    None,
                    a=model_diff,
                    b=golden_diff,
                    autojunk=False,
                ).ratio()
                scores.append(score)

            return dict(score=sum(scores) / len(scores), verification_result_info=dict())

        except Exception as e:
            return dict(
                score=0,
                verification_result_info={"failure_reason": f"Error in scoring patches: {e}"},
            )

    def verify(self, result: Response):
        """
        Evaluates the code patches by comparing the model's patches against golden patches.

        The score lies between 0 and 1, representing the similarity between the model's patches and the golden patches.
        """
        print("Processing example: ", result["problem_id"])
        verification_info = result["verification_info"]
        json_output = parse_json_codeblock_from_model_output(result["llm_response"])
        return self.score_patching(verification_info, json_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SWE-Fixer patches")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file containing patches to verify")
    args = parser.parse_args()

    to_verify = []
    with open(args.file, "r") as f:
        for line in f:
            d = json.loads(line)
            d["verification_info"] = ast.literal_eval(d["verification_info"])
            d["metadata"] = ast.literal_eval(d["metadata"])
            to_verify.append(d)

    verifier = SweFixerVerifier()
    for item in to_verify:
        result = verifier.verify(item)
        print(result)
