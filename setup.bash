#!/usr/bin/env bash

# Setup Script for ASTraM Traffic Event Intelligence
# Team Insight.exe

echo "Initializing environment setup..."

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv package manager not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv package manager detected."
fi

# Sync dependencies using pyproject.toml
echo "Synchronizing project libraries..."
uv sync

echo "Starting Streamlit Dashboard..."
uv run streamlit run streamlit_app/app.py
