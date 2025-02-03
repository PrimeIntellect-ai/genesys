<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

<h3 align="center">
GENESYS: Reasoning Data Generation & Verification
</h3>
<p align="center">
| <a href=""><b>Blog</b></a> | <a href=""><b>X Thread</b></a> | <a href=""><b>SYNTHETIC-1 Dashboard</b></a> |
</p>

---


Genesys is a library for synthetic reasoning data generation and verification, used to generate [SYNTHETIC-1]().

The library has two main entrypoints: 
- `src/genesys/generate.py` is used to sample responses to tasks from a given dataset using a teacher model.
- `src/genesys/verify.py` is used to verify responses and assign rewards using verifiers. Verifiers are task-dependent - some task responses are verified by executing code tests, others are verified with an LLM judge. Genesys is designed to make it easy to add custom verifiers for new tasks.

# Usage

### Installation

**Quick Install:** Run the following command for a quick install:
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/genesys/main/scripts/install/install.sh | bash
```

### Data Generation

To check that your installation has succeeded, you can run the following command to generate data with a small model:

```
# with config file; see /configs for all configurations
uv run python src/genesys/generate.py @ configs/debug.toml

# otherwise, with --flags
uv run python src/genesys/generate.py \
  --name_model "Qwen/Qwen2.5-Coder-0.5B" \
  --num_gpus 1 \
  --sample_per_file 8 \
  --temperature 0.6 \
  --max_tokens 16384 \
  --data.max_samples 16 \
  --data.batch_size 8 \
  --data.path "PrimeIntellect/verifiable-math" # Follow the schema from "PrimeIntellect/verifiable-math", "PrimeIntellect/verifiable-coding", etc.
```

Your file with responses will be saved to `/output`.

**Run with Docker:** You can also generate data using the docker image:

```
sudo docker run --gpus all  -it primeintellect/genesys:latest uv run python src/genesys/generate.py @ configs/debug.toml
```

### Verification

To verify model responses, you can use the `src/genesys/verify.py` script along with the output file from `src/genesys/generate.py` located in `output`.

```
uv run python src/genesys/verify.py --file <path-to-out-file> # output file is usually at /output/out_<some_uuid>.jsonl
```

The verification loop runs asynchronously to parallelize verification and speed up processing. .

### Adding Tasks & Verifiers

You might want to run your own
### running test

```
uv run pytest
```


## Docker


build 

```
sudo docker build -t primeintellect/genesys:latest .
```

run 

```
sudo docker run --gpus all  -it primeintellect/genesys:latest uv run python src/genesys/generate.py @ configs/debug.toml
```


