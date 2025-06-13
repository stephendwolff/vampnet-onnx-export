# Contributing to VampNet ONNX Export

Thank you for your interest in contributing to the VampNet ONNX Export project! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/vampnet-onnx-export.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/

# Check code style
flake8 src/ tests/
```

## Priority Areas

We especially welcome contributions in these areas:

### 1. Dynamic Sequence Length Support
The transformer currently requires fixed 100-token sequences. We need:
- Custom attention implementation that supports dynamic shapes
- ONNX operator modifications to handle variable-length sequences
- Automatic sequence length detection and padding

### 2. Full DAC Codec Export
The current implementation uses simplified codecs. We need:
- Extract actual DAC codec weights from VampNet
- Implement complete encode/decode operations
- Maintain compatibility with original model outputs

### 3. Sampling and Temperature Support
Current export uses deterministic argmax. We need:
- Implement proper sampling in ONNX-compatible way
- Add temperature scaling
- Support for top-k and top-p sampling

### 4. Platform-Specific Optimizations
- TensorRT optimization scripts
- Core ML conversion examples
- Mobile deployment guides
- WebAssembly builds

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep line length under 88 characters (Black default)

Example:
```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    normalize: bool = True
) -> np.ndarray:
    """
    Process audio for VampNet input.
    
    Args:
        audio: Input audio array [channels, samples]
        sample_rate: Audio sample rate in Hz
        normalize: Whether to normalize audio
        
    Returns:
        Processed audio array [1, samples]
    """
    # Implementation here
    pass
```

## Testing

All new features should include tests:

```python
# tests/test_your_feature.py
import pytest
import numpy as np
from src.your_module import your_function


def test_your_function():
    """Test basic functionality."""
    input_data = np.random.randn(1, 100)
    result = your_function(input_data)
    assert result.shape == (1, 100)
    assert np.all(np.isfinite(result))


def test_your_function_edge_case():
    """Test edge cases."""
    # Empty input
    with pytest.raises(ValueError):
        your_function(np.array([]))
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all new functions
- Include usage examples in docstrings
- Update notebooks if changing core functionality

## Commit Messages

Use clear, descriptive commit messages:
- `feat: Add dynamic sequence length support for transformer`
- `fix: Correct padding in audio processor`
- `docs: Update README with new export options`
- `test: Add tests for mask generator`
- `perf: Optimize codec encoder for CPU inference`

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add description of changes
4. Link any related issues
5. Request review from maintainers

## Issue Reporting

When reporting issues, please include:
- Python version
- PyTorch version
- ONNX Runtime version
- Operating system
- Complete error messages
- Minimal code to reproduce

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Contact maintainers for sensitive issues

Thank you for contributing!