import docker
from pydantic import BaseModel, Field
from typing import List, Dict
import io
import tarfile
import uuid
import re

class CodeTestsVerification(BaseModel):
    type: str = Field("code_tests")
    language: str
    test_cases: List[Dict]
    
def extract_code(response: str) -> str:
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
    
    if code_blocks:
        return code_blocks[-1].strip()
    else:
        return None

def copy_to_container(container, dst, content):
    data = io.BytesIO()
    with tarfile.TarFile(fileobj=data, mode='w') as tf:
        tar_info = tarfile.TarInfo(name=dst)
        tar_info.size = len(content)
        tf.addfile(tar_info, io.BytesIO(content.encode('utf-8')))

    data.seek(0)
    
    container.put_archive("/", data)
    
def execute_python(code, inputs, docker_client):
    
    code_filename = f"code_{uuid.uuid4().hex}.py"
    input_filename = f"input_{uuid.uuid4().hex}.txt"

    container = docker_client.containers.create(
        image="python:3.9",
        command="sleep infinity", 
        tty=False,
        stdin_open=False
    )
    container.start()

    copy_to_container(container, code_filename, code)
    copy_to_container(container, input_filename, inputs)

    run_cmd = ["sh", "-c", f"python {code_filename} < {input_filename}"]
    run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
    output = run_result.output.decode()

    container.stop()
    container.remove()

    return output

def execute_rust(code, inputs, docker_client):
    code_filename = f"main_{uuid.uuid4().hex}.rs"
    input_filename = f"input_{uuid.uuid4().hex}.txt"

    container = docker_client.containers.create(
        image="rust:latest",
        command="sleep infinity",
        tty=False,
        stdin_open=False
    )
    
    container.start()

    copy_to_container(container, code_filename, code)
    copy_to_container(container, input_filename, inputs)

    compile_cmd = f"rustc {code_filename} -o main"
    compile_result = container.exec_run(cmd=compile_cmd, stdout=True, stderr=True)
    if compile_result.exit_code != 0:
        error_output = compile_result.output.decode()
        container.stop()
        container.remove()
        return f"Compilation Error:\n{error_output}"

    run_cmd = ["sh", "-c", f"./main < {input_filename}"]
    run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
    output = run_result.output.decode()

    container.stop()
    container.remove()

    return output

def execute_cpp(code, inputs, docker_client):
    code_filename = f"main_{uuid.uuid4().hex}.cpp"
    input_filename = f"input_{uuid.uuid4().hex}.txt"

    container = docker_client.containers.create(
        image="gcc:latest",
        command="sleep infinity",
        tty=False,
        stdin_open=False
    )
    container.start()

    copy_to_container(container, code_filename, code)
    copy_to_container(container, input_filename, inputs)

    compile_cmd = f"g++ {code_filename} -o main"
    compile_result = container.exec_run(cmd=compile_cmd, stdout=True, stderr=True)
    if compile_result.exit_code != 0:
        error_output = compile_result.output.decode()
        container.stop()
        container.remove()
        return f"Compilation Error:\n{error_output}"

    run_cmd = ["sh", "-c", f"./main < {input_filename}"]
    run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
    output = run_result.output.decode()

    container.stop()
    container.remove()

    return output
        
def execute_javascript(code, inputs, docker_client):
    code_filename = f"main_{uuid.uuid4().hex}.js"
    input_filename = f"input_{uuid.uuid4().hex}.txt"

    container = docker_client.containers.create(
        image="node:latest",
        command="sleep infinity",
        tty=False,
        stdin_open=False
    )
    container.start()

    copy_to_container(container, code_filename, code)
    copy_to_container(container, input_filename, inputs)

    run_cmd = ["sh", "-c", f"node {code_filename} < {input_filename}"]
    run_result = container.exec_run(cmd=run_cmd, stdout=True, stderr=True)
    output = run_result.output.decode()

    container.stop()
    container.remove()

    return output
    
def verify_code(response: str, test_cases, language):
    code = extract_code(response)
    if code is None:
        return 0
        
    docker_client = docker.from_env()
    
    language_executors = {
        "Python": execute_python,
        "Rust": execute_rust,
        "C++": execute_cpp,
        "Javascript": execute_javascript,
    }
    
    executor = language_executors.get(language)
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for test in test_cases:
        output = executor(code, test["input"], docker_client)
        print("output", output)
        print("input", test["input"])
        if output.strip() == test["output"].strip():
            passed_tests += 1
    
    return passed_tests/total_tests    
        
    
    
if __name__ == '__main__':
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
        """
    ]
        
    test_cases = [
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}],
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}],
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}],
        [{"input": "3\n2 2 3\n4 3 7\n10 1 9\n", "output": "1\n6\n-1\n"}]
    ]
    
    languages = ["Python", "C++", "Rust", "Javascript"]
    
    for c, t, l in zip(code_samples, test_cases, languages):
        print(f"\n\n### Testing for {l} ###")
        score = verify_code(c, t, l)
        print(score)
    
    

