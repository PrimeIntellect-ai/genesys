import docker
import io
import tarfile
import uuid
import re
import ast
from typing import List, Dict
from pydantic import BaseModel, Field

class CodeTestsVerification(BaseModel):
    language: str
    test_cases: List[Dict]


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

CONTAINERS = {}

def init_containers():
    """
    call this before tests; we want global containers to not have to restart them for every test
    """
    docker_client = docker.from_env()

    CONTAINERS["python"] = docker_client.containers.run("python:3.9", command="sleep infinity", detach=True)

    CONTAINERS["rust"] = docker_client.containers.run("rust:latest", command="sleep infinity", detach=True)

    CONTAINERS["cpp"] = docker_client.containers.run("gcc:latest", command="sleep infinity", detach=True)

    CONTAINERS["javascript"] = docker_client.containers.run("node:latest", command="sleep infinity", detach=True)


def close_containers():
    for lang, container in CONTAINERS.items():
        container.stop()
        container.remove()
    CONTAINERS.clear()


def copy_to_container(container, dst, content: str):
    data = io.BytesIO()
    with tarfile.TarFile(fileobj=data, mode="w") as tf:
        tar_info = tarfile.TarInfo(name=dst)
        tar_info.size = len(content)
        tf.addfile(tar_info, io.BytesIO(content.encode("utf-8")))

    data.seek(0)
    container.put_archive("/", data)


def extract_code(response: str) -> str:
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    else:
        return None
    

def detect_callable_name(user_code: str, fn_name: str):
    """
    Returns (class_name, is_method) if we find 'fn_name' inside a class.
    Otherwise returns (function_name, is_method=False) if we find 'fn_name' top-level.
    Returns (None, None) if not found at all.
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


def verify_compiled_code(container, code, test_cases, language):
    if language == "cpp":
        source_filename = f"main_{uuid.uuid4().hex}.cpp"
        compile_cmd = f"g++ {source_filename} -o main"
        run_binary = "./main"
    elif language == "rust":
        source_filename = f"main_{uuid.uuid4().hex}.rs"
        compile_cmd = f"rustc {source_filename} -o main"
        run_binary = "./main"
    else:
        return 0.0

    copy_to_container(container, source_filename, code)

    compile_result = container.exec_run(cmd=compile_cmd, stdout=True, stderr=True)
    if compile_result.exit_code != 0:
        error_output = compile_result.output.decode()
        print("Compilation Error:\n", error_output)
        return 0.0

    passed_tests = 0
    total_tests = len(test_cases)

    for test in test_cases:
        input_filename = f"input_{uuid.uuid4().hex}.txt"
        copy_to_container(container, input_filename, test["input"])

        run_cmd = ["sh", "-c", f"{run_binary} < {input_filename}"]
        run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
        output = run_result.output.decode()
        
        normalized_output = '\n'.join(line.strip() for line in output.strip().split('\n'))
        normalized_expected = '\n'.join(line.strip() for line in test["output"].strip().split('\n'))

        if normalized_output == normalized_expected:

            passed_tests += 1

    return passed_tests / total_tests


def verify_python_call_based_code(container, user_code: str, test_cases: List[Dict]):
    """
    test_cases look like:
    [
      {
         "fn_name": "computeSum",
         "input": [3,5],
         "output": 8
      },
      ...
    ]
    The llm generated code might contain a class with a method:
      class Solution:
          def computeSum(self, x, y): return x+y
    or a function
      def computeSum(x, y): return x+y
    """

    injection = "from typing import List, Dict, Set, Tuple, Union, Optional\n" # often starter has type hints
    user_code = injection + user_code

    passed_tests = 0
    total_tests = len(test_cases)

    code_filename = f"user_code_{uuid.uuid4().hex}.py"
    copy_to_container(container, code_filename, user_code)

    for test in test_cases:
        fn_name = test["fn_name"]
        args = test["input"]
        expected = test["output"]

        class_or_func, is_method = detect_callable_name(user_code, fn_name)
                
        if class_or_func is None:
            continue

        if is_method:
            harness_code = HARNESS_CODE_METHOD.format(
                code_filename=code_filename[:-3],
                class_or_func=class_or_func,
                fn_name=fn_name,
                args=args
            )
            
        else:
            harness_code = HARNESS_CODE_FUNCTION.format(
                code_filename=code_filename[:-3],
                fn_name=fn_name,
                args=args
            )

        harness_filename = f"harness_{uuid.uuid4().hex}.py"
        copy_to_container(container, harness_filename, harness_code)

        run_result = container.exec_run(["python", harness_filename], stdout=True, stderr=True)
        output = run_result.output.decode().strip()
        
        if output == str(expected).strip():
            passed_tests += 1

    return passed_tests / total_tests


def verify_interpreted_code(container, code, test_cases, language):
    """
    Copy code once, then run multiple times with different inputs.
    """
    if language == "python":
        code_filename = f"code_{uuid.uuid4().hex}.py"
        run_cmd_template = "python {code_file} < {input_file}"
    elif language == "javascript":
        code_filename = f"main_{uuid.uuid4().hex}.js"
        run_cmd_template = "node {code_file} < {input_file}"
    else:
        return 0.0

    copy_to_container(container, code_filename, code)

    passed_tests = 0
    total_tests = len(test_cases)

    for test in test_cases:
        input_filename = f"input_{uuid.uuid4().hex}.txt"
        copy_to_container(container, input_filename, test["input"])

        run_cmd_str = run_cmd_template.format(code_file=code_filename, input_file=input_filename)
        run_cmd = ["sh", "-c", run_cmd_str]
        run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
        output = run_result.output.decode()
        
        normalized_output = '\n'.join(line.strip() for line in output.strip().split('\n'))
        normalized_expected = '\n'.join(line.strip() for line in test["output"].strip().split('\n'))

        if normalized_output == normalized_expected:
            passed_tests += 1

    return passed_tests / total_tests


def verify_code(response: str, test_cases, language):
    code = extract_code(response)
    if code is None:
        return 0.0

    if language not in CONTAINERS:
        print(f"No container found for language: {language}")
        return 0.0

    container = CONTAINERS[language]

    if all(tc.get("type") == "function_call" for tc in test_cases):
        if language == "python":
            return verify_python_call_based_code(container, code, test_cases)
        else:
            print("Call-based code testing not implemented for this language.")
            return 0.0
    else:
        if language in ["cpp", "rust"]:
            return verify_compiled_code(container, code, test_cases, language)
        elif language in ["python", "javascript"]:
            return verify_interpreted_code(container, code, test_cases, language)
        else:
            print("Unsupported language:", language)
            return 0.0


if __name__ == "__main__":
    init_containers()

    code_samples = [
        """
Here's a python solution to the problem:

```python
q = int(input())

for _ in range(q):
    x, y, k = map(int, input().split())
    x, y = abs(x), abs(y)
    x, y = max(x, y), min(x, y)
    
    if x % 2 != k % 2:
        k -= 1
    
    y -= 1
    
    if x > k:
        print(-1)
        continue
    
    if (x - y) % 2:
        k -= 1
        x -= 1
    
    print(k)
```

This solution handles the given constraints and should work efficiently.
        """,
        """
I've implemented a cpp solution for the problem. Here's the code:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    cin >> q;

    while (q--) {
        long long x, y, k;
        cin >> x >> y >> k;

        // 1. Convert x, y to absolute values
        x = llabs(x);
        y = llabs(y);

        // 2. Ensure x >= y
        if (x < y) {
            long long temp = x;
            x = y;
            y = temp;
        }

        // 3. Adjust k if parity differs
        if ((x % 2) != (k % 2)) {
            k -= 1;
        }

        // 4. Decrement y by 1
        y -= 1;

        // 5. If x > k, print -1
        if (x > k) {
            cout << -1 << \"\\n\";
            continue;
        }

        // 6. Adjust if (x - y) is odd
        if ((x - y) % 2 != 0) {
            k -= 1;
            x -= 1;
        }

        // 7. Print k
        cout << k << \"\\n\";
    }

    return 0;
}
```

This cpp implementation should solve the problem efficiently.
        """,
        """
For the given problem, here's a rust implementation that should work:

```rust
use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // Read number of queries q
    let q: i32 = lines.next().unwrap().unwrap().parse().unwrap();

    for _ in 0..q {
        let line = lines.next().unwrap().unwrap();
        let mut parts = line.split_whitespace();

        // Read x, y, k
        let mut x: i64 = parts.next().unwrap().parse().unwrap();
        let mut y: i64 = parts.next().unwrap().parse().unwrap();
        let mut k: i64 = parts.next().unwrap().parse().unwrap();

        // 1. Convert x, y to absolute values
        x = x.abs();
        y = y.abs();

        // 2. Ensure x >= y
        if x < y {
            let temp = x;
            x = y;
            y = temp;
        }

        // 3. Adjust k if parity differs
        if (x % 2) != (k % 2) {
            k -= 1;
        }

        // 4. Decrement y by 1
        y -= 1;

        // 5. If x > k, print -1
        if x > k {
            println!("-1");
            continue;
        }

        // 6. Adjust if (x - y) is odd
        if (x - y) % 2 != 0 {
            k -= 1;
            x -= 1;
        }

        // 7. Print k
        println!("{}", k);
    }
}
```

This rust code should solve the problem efficiently while handling all the given constraints.
        """,
        """
I've created a JavaScript solution for the problem. Here's the code:

```javascript
const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

let inputLines = [];
rl.on('line', (line) => {
    inputLines.push(line.trim());
});
rl.on('close', () => {
    let q = parseInt(inputLines[0], 10);
    let idx = 1;

    for (let i = 0; i < q; i++) {
    let [x, y, k] = inputLines[idx++].split(' ').map(Number);

    // 1. Convert x, y to absolute values
    x = Math.abs(x);
    y = Math.abs(y);

    // 2. Ensure x >= y
    if (x < y) {
        [x, y] = [y, x];
    }

    // 3. Adjust k if parity differs
    if ((x % 2) !== (k % 2)) {
        k--;
    }

    // 4. Decrement y by 1
    y--;

    // 5. If x > k, print -1
    if (x > k) {
        console.log(-1);
        continue;
    }

    // 6. Adjust if (x - y) is odd
    if ((x - y) % 2 !== 0) {
        k--;
        x--;
    }

    // 7. Print k
    console.log(k);
    }
});
```

This JavaScript implementation should work correctly for the given problem.
        """,
    ] * 2

    test_cases = [
        [{"type": "stdin_stdout", "input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 2,
        [{"type": "stdin_stdout", "input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 2,
        [{"type": "stdin_stdout", "input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 2,
        [{"type": "stdin_stdout", "input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 2,
    ] * 10

    languages = ["python", "cpp", "rust", "javascript"] * 2
    
    
    call_based_codes = [
        """
Here's my solution

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        
        return candidate
```

Hope it helps
        """,
        """
Here's my solution

```python
def majorityElement(nums: List[int]) -> int:
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    return candidate
```

Hope it helps
        """
        
    ]
    
    call_based_test_cases = [
        [{"type": "function_call", "fn_name": "majorityElement", "input": [[3, 2, 3]], "output": 3}],
        [{"type": "function_call", "fn_name": "majorityElement", "input": [[3, 2, 3]], "output": 3}]
    ]
    
    call_based_languages = [
        "python",
        "python"
    ]
    
    all_codes = code_samples +  call_based_codes
    all_test_cases = test_cases + call_based_test_cases
    all_languages = languages + call_based_languages

    import time

    start = time.time()
    for code, test, lang in zip(all_codes, all_test_cases, all_languages):
        print(f"\n\n### Testing for {lang} ###")
        score = verify_code(code, test, lang)
        print(score)
    end = time.time()

    print("time", end - start)

    close_containers()
