import docker
import io
import tarfile
import uuid
import re
import ast
from genesys.verifiers.base_verifier import BaseVerifier
from typing import List, Dict

HARNESS_CODE_METHOD = """
import {code_filename} as user_code
obj = user_code.{class_or_func}()
method = getattr(obj, '{fn_name}')
args = {args}
res = method(*args)
print(res)
"""

HARNESS_CODE_FUNCTION = """
import {code_filename} as user_code
method = getattr(user_code, '{fn_name}')
args = {args}
res = method(*args)
print(res)
"""


class CodeVerifier(BaseVerifier):
    def __init__(self):
        """
        Constructor: initializes a Docker client and starts one container
        per language. This happens as soon as the CodeVerifier is created.
        """
        self.docker_client = docker.from_env()
        self.containers = {}
        self._init_containers()
        self.timeout = 30
        self.max_parallel = 5

    def __del__(self):
        """
        Destructor: attempts to close containers when this object is garbage-collected.
        NOTE: In Python, it can be safer to call an explicit close() method.
        """
        self._close_containers()

    def _init_containers(self):
        """
        Start one Docker container per supported language. They remain running
        (sleep infinity) so we can quickly exec into them multiple times.
        """
        self.containers["python"] = self.docker_client.containers.run(
            "python:3.9", command="sleep infinity", detach=True
        )
        self.containers["rust"] = self.docker_client.containers.run(
            "rust:latest", command="sleep infinity", detach=True
        )
        self.containers["cpp"] = self.docker_client.containers.run("gcc:latest", command="sleep infinity", detach=True)
        self.containers["javascript"] = self.docker_client.containers.run(
            "node:latest", command="sleep infinity", detach=True
        )

    def _close_containers(self):
        """
        Stop and remove all running containers we created.
        """
        for container in self.containers.values():
            try:
                container.stop()
                container.remove()
            except Exception as e:
                print("Error while stopping/removing container:", e)
        self.containers.clear()

    def _copy_to_container(self, container, dst, content):
        """
        Copy text content into a given path 'dst' in the container.
        """
        data = io.BytesIO()
        with tarfile.TarFile(fileobj=data, mode="w") as tf:
            tar_info = tarfile.TarInfo(name=dst)
            tar_info.size = len(content)
            tf.addfile(tar_info, io.BytesIO(content.encode("utf-8")))

        data.seek(0)
        container.put_archive("/", data)

    def _extract_code(self, response: str) -> str:
        """
        Extract the last fenced code block (triple backtick) from a markdown-like response.
        Returns None if no code block is found.
        """
        if not isinstance(response, str):
            print("Code response is not str:", response)
            return None

        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        else:
            return None

    def _detect_callable_name(self, user_code: str, fn_name: str):
        """
        Returns (class_name, True) if we find 'fn_name' as a method in a class,
        (fn_name, False) if we find 'fn_name' as a top-level function,
        or (None, None) if not found.
        """
        try:
            tree = ast.parse(user_code)
        except SyntaxError:
            return (None, None)

        found_class = None
        found_function_top_level = False

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef) and subnode.name == fn_name:
                        found_class = node.name
                        break
                if found_class:
                    break
            elif isinstance(node, ast.FunctionDef) and node.name == fn_name:
                found_function_top_level = True

        if found_class:
            return (found_class, True)
        elif found_function_top_level:
            return (fn_name, False)

        return (None, None)

    def _verify_compiled_code(self, container, code: str, test_cases: List[Dict], language: str) -> float:
        """
        For compiled languages (C++/Rust), copy code, compile, then run the binary with test inputs.
        Return fraction of test cases passed.
        """
        if language == "cpp":
            source_filename = f"main_{uuid.uuid4().hex}.cpp"
            compile_cmd = f"g++ {source_filename} -o main"
            run_binary = "./main"
        elif language == "rust":
            source_filename = f"main_{uuid.uuid4().hex}.rs"
            compile_cmd = f"rustc {source_filename} -o main"
            run_binary = "./main"
        else:
            raise ValueError(f"The language {language} is not supported")

        self._copy_to_container(container, source_filename, code)

        compile_result = container.exec_run(cmd=compile_cmd, stdout=True, stderr=True)
        if compile_result.exit_code != 0:
            error_output = compile_result.output.decode()
            print("Compilation Error:\n", error_output)
            return dict(score=0.0, verification_result_info=dict(failure_reason="compilation_error"))

        passed_tests = 0
        total_tests = len(test_cases)

        for test in test_cases:
            input_filename = f"input_{uuid.uuid4().hex}.txt"
            self._copy_to_container(container, input_filename, str(test["input"]))

            run_cmd = ["sh", "-c", f"{run_binary} < {input_filename}"]
            run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
            output = run_result.output.decode()

            normalized_output = "\n".join(line.strip() for line in output.strip().split("\n"))
            normalized_expected = "\n".join(line.strip() for line in str(test["output"]).strip().split("\n"))

            if normalized_output == normalized_expected:
                passed_tests += 1

        return dict(score=passed_tests / total_tests, verification_result_info=dict())

    def _verify_python_call_based_code(self, container, user_code: str, test_cases: List[Dict]) -> float:
        """
        For Python code that is expected to define a function or a method in a class.
        Each test case has 'fn_name', 'input' (list of args), and 'output' (expected).
        """
        # Provide some standard type imports before the user code
        injection = "from typing import List, Dict, Set, Tuple, Union, Optional\n"
        user_code = injection + user_code

        passed_tests = 0
        total_tests = len(test_cases)

        code_filename = f"user_code_{uuid.uuid4().hex}.py"
        self._copy_to_container(container, code_filename, user_code)

        for test in test_cases:
            fn_name = test["fn_name"]
            args = test["input"]
            expected = test["output"]

            class_or_func, is_method = self._detect_callable_name(user_code, fn_name)
            if class_or_func is None:
                # The code didn't define the requested function at all
                continue

            if is_method:
                harness_code = HARNESS_CODE_METHOD.format(
                    code_filename=code_filename[:-3], class_or_func=class_or_func, fn_name=fn_name, args=args
                )
            else:
                harness_code = HARNESS_CODE_FUNCTION.format(
                    code_filename=code_filename[:-3], fn_name=fn_name, args=args
                )

            harness_filename = f"harness_{uuid.uuid4().hex}.py"
            self._copy_to_container(container, harness_filename, harness_code)

            run_result = container.exec_run(["python", harness_filename], stdout=True, stderr=True)
            output = run_result.output.decode().strip()

            if output == str(expected).strip():
                passed_tests += 1

        return dict(score=passed_tests / total_tests, verification_result_info=dict())

    def _verify_interpreted_code(self, container, code: str, test_cases: List[Dict], language: str) -> float:
        """
        For interpreted languages (Python/Node). We copy the code once, then run it for each test.
        """
        if language == "python":
            code_filename = f"code_{uuid.uuid4().hex}.py"
            run_cmd_template = "python {code_file} < {input_file}"
        elif language == "javascript":
            code_filename = f"main_{uuid.uuid4().hex}.js"
            run_cmd_template = "node {code_file} < {input_file}"
        else:
            raise ValueError(f"language {language} not supported")

        self._copy_to_container(container, code_filename, code)

        passed_tests = 0
        total_tests = len(test_cases)

        for test in test_cases:
            input_filename = f"input_{uuid.uuid4().hex}.txt"
            self._copy_to_container(container, input_filename, str(test["input"]))

            run_cmd_str = run_cmd_template.format(code_file=code_filename, input_file=input_filename)
            run_cmd = ["sh", "-c", run_cmd_str]
            run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
            output = run_result.output.decode()

            normalized_output = "\n".join(line.strip() for line in output.strip().split("\n"))
            normalized_expected = "\n".join(line.strip() for line in str(test["output"]).strip().split("\n"))

            if normalized_output == normalized_expected:
                passed_tests += 1

        return dict(score=passed_tests / total_tests, verification_result_info=dict())

    def verify(self, result: Dict) -> float:
        """
        The main entry point to verify code. Extracts the code from the LLM response,
        looks up the Docker container for the given language, then tests it according
        to the test_cases. Returns a fraction [0..1].
        """
        response = result["llm_response"]
        test_cases = result["verification_info"]["test_cases"]
        language = result["verification_info"]["language"]

        code = self._extract_code(response)
        if code is None:
            return dict(score=0.0, verification_result_info=dict(failure_reason="no_code_in_response"))

        if language not in self.containers:
            raise ValueError(f"No container found for language: {language}")

        container = self.containers[language]

        # If all test cases are "function_call", we handle them differently for Python
        if all(tc.get("type") == "function_call" for tc in test_cases):
            if language == "python":
                return self._verify_python_call_based_code(container, code, test_cases)
            else:
                raise ValueError(f"Call-based code testing not implemented for {language}.")
        else:
            # Otherwise, use the standard compile-or-run approach
            if language in ["cpp", "rust"]:
                return self._verify_compiled_code(container, code, test_cases, language)
            elif language in ["python", "javascript"]:
                return self._verify_interpreted_code(container, code, test_cases, language)
            else:
                raise ValueError("Unsupported language:", language)
