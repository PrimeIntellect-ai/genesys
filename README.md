<p align="center">
<h2 align="center">INTELLECT-MATH: Frontier Mathematical Reasoning through Better Initializations for Reinforcement Learning</h1>
</p>

<img width="363" alt="Screenshot 2025-02-03 at 03 15 04" src="https://github.com/user-attachments/assets/9deed6d6-d9d7-40a0-b163-12ab8579f450" />


<p align="center">
Test
</p>
<p>
  
</p>



## install

quick install (only will work when repo got open source)
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/genesys/main/scripts/install/install.sh | bash
```

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


