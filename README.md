# Genesys


## install

```
git clone git@github.com:PrimeIntellect-ai/genesys.git
uv sync --extra sglang
```

## run

This is a short run to test if the repo is installed correctly

```
uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 2
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
sudo docker build -t genesys:latest .
```

run 

```
sudo docker run --gpus all  -it genesys:latest uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 2
```


