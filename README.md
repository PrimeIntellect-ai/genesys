# intellect


## install

```
git clone git@github.com:PrimeIntellect-ai/intellect.git
uv sync --extra sglang
```

## run

This is a short run to test if the repo is installed correctly

```
uv run python src/intellect/generate.py --model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8
```

for dev setup:

```
uv run pre-commit install
```


### running test

```
uv run pytest
```



