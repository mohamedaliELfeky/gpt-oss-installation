#!/bin/bash
set -e  # exit on first error

echo "Setting up GPT-OSS-20B model server..."

# Update system
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install dependencies
echo "Installing system dependencies..."
sudo apt-get install -y curl git python3.12 python3.12-venv python3.12-dev build-essential

# Install uv package manager
echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment with uv
echo "Creating virtual environment..."
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vllm with CUDA 12.8 support
echo "Installing vLLM with GPU support..."
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

echo "Setup complete! Starting model server on port 8000..."
echo "The model will be available at: http://localhost:8000/v1"
echo "Press Ctrl+C to stop the server"

# Serve the 20B model on port 8000
vllm serve openai/gpt-oss-20b --port 8000