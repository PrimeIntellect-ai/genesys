<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<h3 align="center">
GENESYS: Reasoning Data Generation & Verification
</h3>
<p align="center">
| <a href="https://www.primeintellect.ai/blog/synthetic-1"><b>Blog</b></a> | <a href="https://x.com/PrimeIntellect/status/1887635142644277692"><b>X Thread</b></a> | <a href="https://app.primeintellect.ai/intelligence/synthetic-1"><b>SYNTHETIC-1 Dashboard</b></a> |
</p>

---


Genesys is a library for synthetic reasoning data generation and verification, used to generate [SYNTHETIC-1](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37).

The library has two main entrypoints: 
- `src/genesys/generate.py` is used to sample responses to tasks from a given dataset using a teacher model.
- `src/genesys/verify.py` is used to verify responses and assign rewards using verifiers


# Usage

## Installation

**Quick Install:** Run the following command for a quick install:


```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/genesys/refs/heads/main/script/install.sh | bash
```

or follow this steps to install manually

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

then

```bash
git clone git@github.com:PrimeIntellect-ai/genesys.git
cd genesys
uv venv --python 3.12
uv sync --extra sglang
```


## Data Generation

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

## prod run

```
export HF_HUB_ETAG_TIMEOUT=500,
export GCP_CREDENTIALS_BASE64=XXX
bash ./script/entrypoint.sh
```

## Verification

To verify model responses, you can use the `src/genesys/verify.py` script along with the output file from `src/genesys/generate.py` located in `output`.

```
uv run python src/genesys/verify.py --file <path-to-out-file> # output file is usually at /output/out_<some_uuid>.jsonl
```

The verification loop runs asynchronously to parallelize verification and speed up processing. .

## Adding your own Tasks & Verifiers

Genesys is built to be easily extendable for your own tasks & verifiers. You can generate responses for your own data by using a Huggingface dataset with our schema and add your own verifier with minimal code.

### Using your own Data 

To generate data from your own tasks using `src/genesys/generate.py`, you should pass a huggingface dataset with the same schema as `PrimeIntellect/verifiable-math` or others. This is what the schema looks like:

```python
class Task(BaseModel):
    problem_id: str
    source: str # source of the dataset
    task_type: str # this will be used to map the response to a verifier
    in_source_id: Optional[str] 
    prompt: str
    gold_standard_solution: Optional[str]
    verification_info: Dict # dict is empty if no data needed
    metadata: Dict # dict is empty if no data needed
```

The output from the generation script is a `.jsonl` file with each line containing a `Response` object:

```python
class Response(BaseModel):
    problem_id: str
    source: str
    task_type: str
    in_source_id: Optional[str]
    prompt: str
    gold_standard_solution: Optional[str]
    verification_info: Dict 
    metadata: Dict
    llm_response: str # llm response string
    response_id: str
    model_name: str
    generation_config: Dict # sampling parameters
    machine_info: Dict
```

### Adding a Verifier

To implement a verifier, you have to 1) add a verifier class implementing a `verify` function that receives a `Response` object and 2) add the verifier to a verifier registry.

You can implement your own verifier in `src/genesys/verifiers`:
```python
from genesys.schemas import Response
from genesys.verifiers.base_verifier import BaseVerifier

class LengthVerifier(BaseVerifier):      
    max_parallel = 30 # how many times the task should be executed in parallel max - relevant when doing LLM calls with rate limits
    timeout = 10  # timeout in seconds - when running LLM generated code, we want to avoid it getting stuck in an infinite loop

    # optional: __init__ function for set up (e.g. LLM API client)

    def verify(self, result: Response):
        """
        Required: this example verify function checks the length of the llm response
        and rewards responses over a threshold specified in the dataset.

        The output should be a dict with a score from 0-1 and a 'verification_result_info' dict containing metadata
        """
        response = result["llm_response"]
        threshold_small = result["verification_info"]["length_threshold_small"]
        threshold_large = result["verification_info"]["ength_threshold_large"]

        if len(response) > threshold_large:
            score = 1.0
        elif len(response) > threshold_small:
            score = 0.5
        else:
            score = 0.0

        return dict(score=score, verification_result_info={}) # add metadata if needed in verification_result_info

    # optional: shutdown() function for termination, gets called after all responses are verified. For instance, we use this function in the code verifier to shut down docker images

```

Now, add your verifier to the verifier registry in `src/genesys/verifiers/registry.py`:
```python
from genesys.verifiers.code_test_verifier import CodeVerifier
from genesys.verifiers.math_verifier import MathVerifier
from genesys.verifiers.llm_judge_verifier import LlmJudgeVerifier
from genesys.verifiers.code_output_prediction_verifier import CodeUnderstandingVerifier
from genesys.verifiers.length_verifier import LengthVerifier # your verifier

VERIFIER_REGISTRY = {
    "verifiable_code": CodeVerifier,
    "verifiable_math": MathVerifier,
    "llm_judgeable_groundtruth_similarity": LlmJudgeVerifier,
    "verifiable_code_understanding": CodeUnderstandingVerifier,
    "length_adherance": LengthVerifier
}
```

Every task from your dataset with `"task_type": "length_adherance"` will now be verified with the implemented length verifier when running `src/genesys/verify.py`.


## Pushing data to gcp bucket

For pushing the data to s3/gcp bucket, you have to download a service account key file with the permission to push to your bucket, encode it to base64 and set the encoded file as `GCP_CREDENTIALS_BASE64`. Then you can specify your bucket via the `--gcp_bucket` flag

```
export GCP_CREDENTIALS_BASE64=$(base64 -w 0 /path/to/your/service-account-key.json)
uv run python src/genesys/generate.py @ configs/debug.toml --gcp_bucket checkpoints_pi/test_data
```
