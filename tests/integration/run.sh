#!/bin/bash

# Exit in case of error
set -e

# Change to the integration tests directory
cd "$(dirname "$0")/tests/integration"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Use `.venv\Scripts\activate` on Windows

# Upgrade pip
pip install --upgrade pip

# Install your package from Test PyPI
pip install --index-url https://test.pypi.org/simple/ your-package-name

# Optionally, install any other necessary dependencies that might not be available on Test PyPI
# For example, if your package depends on libraries that are only on the main PyPI:
pip install some-dependency

# Run pytest
pytest

# Deactivate the virtual environment
deactivate
