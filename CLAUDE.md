# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VampNet ONNX Export is a toolkit for converting VampNet (masked acoustic token modeling for music generation) to ONNX format for cross-platform deployment. The project exports VampNet's components (codec, transformer, mask generator) to ONNX for running on mobile, web, and edge devices.

**Requirements**: Python 3.11+ (uses stephendwolff fork of VampNet for Python 3.11 compatibility)

## Architecture

The system consists of modular components that can be exported and used independently:

1. **Audio Processor** (`vampnet_onnx/audio_processor.py`): Handles resampling, normalization, padding
2. **Codec** (`vampnet_onnx/vampnet_codec.py`): 
   - Encoder: Audio → 14 codebooks of discrete tokens (pre-padded version achieves 97.9% token match)
   - Decoder: Tokens → Audio (uses LAC codec, not full DAC)
3. **Mask Generator** (`vampnet_onnx/mask_generator_onnx.py`): Creates masking patterns
4. **Transformer** (`vampnet_onnx/transformer_wrapper.py`): Generates tokens for masked positions
5. **Pipeline** (`vampnet_onnx/pipeline.py`): Chains components for end-to-end processing

Custom ONNX operators are in `scripts/custom_ops/`: RMSNorm, FiLM, CodebookEmbedding, MultiheadAttention

## Development Commands

```bash
# Install with development dependencies
pip install -e ".[dev]"              # Development tools (jupyter, matplotlib)
pip install -e ".[optimization]"      # Optimization tools (onnxruntime-tools, pillow)
pip install -e ".[deployment]"        # Deployment tools (coremltools, tensorflow, onnx-tf)

# Run tests
pytest tests/                         # Run all tests
pytest tests/test_audio_processor.py -v  # Run single test file

# Code quality
black vampnet_onnx/ tests/           # Format code
isort vampnet_onnx/ tests/           # Sort imports
flake8 vampnet_onnx/ tests/          # Check code style
mypy vampnet_onnx/                   # Type checking

# Repository maintenance
./cleanup_repository.sh              # Archive old/experimental files
python scripts/cleanup_models_auto.py  # Clean up old models
```

## Python API Usage (New)

The functionality from scripts has been integrated into the main `vampnet_onnx` package:

```python
from vampnet_onnx import (
    # Export functions
    export_audio_processor,
    export_codec_encoder,
    export_codec_decoder,
    export_mask_generator,
    export_complete_transformer,
    export_pretrained_encoder,
    export_all_components,
    
    # Models
    CoarseTransformer,
    C2FTransformer,
    
    # Weight transfer
    WeightTransferManager,
    complete_weight_transfer,
    
    # Embeddings
    extract_and_convert_embeddings,
    load_embeddings_into_model
)

# Export complete model with weight transfer
export_complete_transformer(
    checkpoint_path="path/to/vampnet.pth",
    output_path="coarse_transformer.onnx",
    model_type="coarse",
    codec_path="path/to/codec.pth"
)

# Export all components at once
export_all_components(
    output_dir="onnx_models",
    checkpoint_path="path/to/vampnet.pth",
    codec_path="path/to/codec.pth",
    export_weighted_models=True
)
```

See `examples/complete_export_example.py` for comprehensive usage examples.

## Legacy Script Usage

The original scripts are still available for reference:

- `scripts/transfer_weights_complete_v3.py` - Complete weight transfer implementation
- `scripts/export_complete_model_to_onnx.py` - Current working ONNX export
- `scripts/export_working_encoder.py` - Pre-padded encoder export (97.9% accuracy)
- `scripts/extract_codec_embeddings.py` - Codec embedding extraction

## Current Limitations

1. **Fixed Sequence Length**: Transformer requires exactly 100 tokens (cannot handle variable lengths)
2. **Simplified Codec**: Uses LAC instead of full DAC codec
3. **No Dynamic Sampling**: Temperature/top-k sampling not yet implemented in ONNX
4. **Audio Padding**: Must be padded to multiples of 768 samples
5. **Output Classes**: VampNet uses 4096 classes vs ONNX's 1025

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_audio_processor.py -v
pytest tests/test_encoder.py -v
pytest tests/test_transformer.py -v
pytest tests/test_decoder.py -v
pytest tests/test_weight_transfer.py -v
pytest tests/test_embeddings.py -v

# Run with coverage
pytest tests/ --cov=vampnet_onnx --cov-report=html
```

### Integration Testing
```python
# Test exported models
python scripts/test_complete_model.py

# Run audio generation pipeline
python scripts/vampnet_full_pipeline_demo.py input.wav output.wav
python generate_audio_simple.py       # Simplified pipeline test

# Benchmark performance
python examples/benchmark_performance.py
```

## Important Notes

- **Weight transfer is now complete** - V3 scripts achieve full weight transfer from pretrained VampNet models
- **Pre-padded encoder** - Achieves 97.9% token match with original VampNet
- **Use V3/V2 exports** - Scripts with "v3", "v2", "fixed", or "proper" suffixes use ONNX-friendly architectures
- **Active model paths**:
  - Coarse: `onnx_models_fixed/coarse_complete_v3.onnx`
  - C2F: `onnx_models_fixed/c2f_complete_v3.onnx`
  - Encoder: `scripts/models/vampnet_encoder_prepadded.onnx`
  - Decoder: `scripts/models/vampnet_codec_decoder.onnx`
- VampNet uses GatedGELU activation and FFN dimension of 2560 (not standard GELU/5120)
- Custom ops require ONNX Runtime 1.15+ for proper execution
- Quantization can reduce model size by 46-55% with minimal quality loss