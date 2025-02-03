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

## Installation

**Quick Install:** Run the following command for a quick install:
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/genesys/main/scripts/install/install.sh | bash
```

The script will execute the following commands:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone git@github.com:PrimeIntellect-ai/genesys.git
cd genesys
uv sync --extra sglang
```

## run

This is a short run to test if the repo is installed correctly

```
uv run python src/genesys/generate.py @ configs/debug.toml
```

For pushing the data to s3/gcp bucket, you have to download a service account key file with the permission to push to your bucket, encode it to base64 and set the encoded file as `GCP_CREDENTIALS_BASE64`. Then you can specify your bucket via the `--gcp_bucket` flag:

```
export GCP_CREDENTIALS_BASE64=$(base64 -w 0 /path/to/your/service-account-key.json)
uv run python src/genesys/generate.py @ configs/debug.toml --gcp_bucket checkpoints_pi/test_data
```

for dev setup:

```
uv run pre-commit install
```


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


