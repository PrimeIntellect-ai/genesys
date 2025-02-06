#!/bin/bash

# Save the original stdout to FD 3
exec 3>&1

# Capture both stdout and stderr while also printing to terminal
output=$(uv run python src/genesys/auto_detect_model_config.py 2>&1 | tee /dev/fd/3)
exit_code=${PIPESTATUS[0]}

if [ $exit_code -ne 0 ]; then
    echo "Error: $output"
    exit $exit_code
fi

# Extract the last line of the output, which should be the model name
model_name=$(echo "$output" | tail -n 1)
echo "The model to run is: $model_name"
uv run python src/genesys/generate.py @ configs/"$model_name".toml