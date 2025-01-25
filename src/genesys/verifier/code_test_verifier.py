import docker
import io
import tarfile
import uuid
import re
from typing import List, Dict
from pydantic import BaseModel, Field


class CodeTestsVerification(BaseModel):
    type: str = Field("code_tests")
    language: str
    test_cases: List[Dict]


# We keep a global dictionary for our containers
CONTAINERS = {}


def init_containers():
    docker_client = docker.from_env()

    CONTAINERS["Python"] = docker_client.containers.run("python:3.9", command="sleep infinity", detach=True)

    CONTAINERS["Rust"] = docker_client.containers.run("rust:latest", command="sleep infinity", detach=True)

    CONTAINERS["C++"] = docker_client.containers.run("gcc:latest", command="sleep infinity", detach=True)

    CONTAINERS["Javascript"] = docker_client.containers.run("node:latest", command="sleep infinity", detach=True)


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


def _verify_compiled_code(container, code, test_cases, language):
    if language == "C++":
        source_filename = f"main_{uuid.uuid4().hex}.cpp"
        compile_cmd = f"g++ {source_filename} -o main"
        run_binary = "./main"
    elif language == "Rust":
        source_filename = f"main_{uuid.uuid4().hex}.rs"
        compile_cmd = f"rustc {source_filename} -o main"
        run_binary = "./main"
    else:
        # Shouldn't happen if we call this only for Rust/C++
        return 0.0

    # Copy source
    copy_to_container(container, source_filename, code)
    # Compile
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

        if output.strip() == test["output"].strip():
            passed_tests += 1

    return passed_tests / total_tests


def _verify_interpreted_code(container, code, test_cases, language):
    """
    Copy code once, then run multiple times with different inputs.
    """
    if language == "Python":
        code_filename = f"code_{uuid.uuid4().hex}.py"
        run_cmd_template = "python {code_file} < {input_file}"
    elif language == "Javascript":
        code_filename = f"main_{uuid.uuid4().hex}.js"
        run_cmd_template = "node {code_file} < {input_file}"
    else:
        return 0.0

    # Copy code once
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

        if output.strip() == test["output"].strip():
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

    if language in ["C++", "Rust"]:
        return _verify_compiled_code(container, code, test_cases, language)
    elif language in ["Python", "Javascript"]:
        return _verify_interpreted_code(container, code, test_cases, language)
    else:
        print("Unsupported language:", language)
        return 0.0


if __name__ == "__main__":
    init_containers()

    code_samples = [
        """
Here's a Python solution to the problem:

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
I've implemented a C++ solution for the problem. Here's the code:

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

This C++ implementation should solve the problem efficiently.
        """,
        """
For the given problem, here's a Rust implementation that should work:

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

This Rust code should solve the problem efficiently while handling all the given constraints.
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
    ] * 10

    test_cases = [
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 10,
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 10,
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 10,
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}] * 10,
    ] * 10

    languages = ["Python", "C++", "Rust", "Javascript"] * 25

    import time

    start = time.time()
    for c, t, l in zip(code_samples, test_cases, languages):
        print(f"\n\n### Testing for {l} ###")
        score = verify_code(c, t, l)
        print(score)
    end = time.time()

    print("time", end - start)

    close_containers()
