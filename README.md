# Genesys


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
git submodule update --init --recursive
uv sync --extra sglang
```

## run

This is a short run to test if the repo is installed correctly

```
uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 8 --sample-per-file 8
```

For pushing the data to s3/gcp bucket you need to have a credentials (.json) downloaded locally and do 



```
export GOOGLE_APPLICATION_CREDENTIALS="/root/creds.json"
uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 8 --sample-per-file 8
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
sudo docker run --gpus all  -it primeintellect/genesys:latest uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 2 --sample-per-file 8

```


