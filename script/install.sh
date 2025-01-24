#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {

    log_info "Cloning repository..."
    git clone https://github.com/PrimeIntellect-ai/genesys.git
    
    log_info "Entering project directory..."
    cd genesys
    
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi
    
    log_info "Creating virtual environment..."
    uv venv
    
    log_info "Activating virtual environment..."
    source .venv/bin/activate
    
    log_info "Installing dependencies..."
    uv sync --extra sglang
        
    log_info "Installation completed! You can double check that everything is install correctly by running 'uv run python src/genesys/generate.py --name_model Qwen/Qwen2.5-Coder-0.5B --num_gpus 1 --batch_size 8 --max_samples 16 --sample_per_file 8'"
}

main