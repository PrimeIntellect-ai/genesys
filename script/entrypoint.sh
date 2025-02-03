#!/bin/bash

# capture both stdout and stderr
exec 3>&1  # save original stdout
output=$(uv run python src/genesys/auto_detect_model_config.py 2>&1 >&3)
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Error: $output"
    exit $exit_code
fi

model_name=$output
echo "The model to run is: $model_name"
uv run python src/genesys/generate.py @ configs/$model_name.toml
