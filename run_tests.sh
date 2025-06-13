#!/bin/bash
# Script to run tests excluding archive folders

echo "Running VampNet ONNX tests (excluding archive folders)..."
echo "=================================================="

# Option 1: Run all tests with the configuration from pyproject.toml
echo -e "\n1. Running with pytest configuration:"
python -m pytest

# Option 2: Run with verbose output
echo -e "\n2. Running with verbose output:"
python -m pytest -v

# Option 3: Run with coverage report
echo -e "\n3. Running with coverage (if pytest-cov is installed):"
python -m pytest --cov=vampnet_onnx --cov-report=term-missing

# Option 4: Run specific test files
echo -e "\n4. Run specific test module example:"
echo "python -m pytest tests/test_audio_processor.py -v"

# Option 5: Run tests matching a pattern
echo -e "\n5. Run tests matching pattern example:"
echo "python -m pytest -k 'encoder' -v"