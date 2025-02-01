import subprocess
import pytest


def test_generate():
    cmd = "python src/genesys/generate.py @ configs/debug.toml".split(" ")

    process = subprocess.Popen(cmd)

    result = process.wait()
    if result != 0:
        pytest.fail(f"Process {result} failed {result}")
