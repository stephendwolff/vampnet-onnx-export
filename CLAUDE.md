# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VampNet ONNX Export is a toolkit for converting VampNet (masked acoustic token modeling for music generation) to ONNX format for cross-platform deployment. The project exports VampNet's components (codec, transformer, mask generator) to ONNX for running on mobile, web, and edge devices.

## Architecture

The system consists of modular components that can be exported and used independently:

1. **Audio Processor** (`vampnet_onnx/audio_processor.py`): Handles resampling, normalization, padding
2. **Codec** (`vampnet_onnx/vampnet_codec.py`): 
   - Encoder: Audio → 14 codebooks of discrete tokens
   - Decoder: Tokens → Audio (uses LAC codec, not full DAC)
3. **Mask Generator** (`vampnet_onnx/mask_generator_onnx.py`): Creates masking patterns
4. **Transformer** (`vampnet_onnx/transformer_wrapper.py`): Generates tokens for masked positions
5. **Pipeline** (`vampnet_onnx/pipeline.py`): Chains components for end-to-end processing

Custom ONNX operators are in `scripts/custom_ops/`: RMSNorm, FiLM, CodebookEmbedding, MultiheadAttention

## Development Commands

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black vampnet_onnx/ tests/

# Check code style
flake8 vampnet_onnx/ tests/

# Clean up old models
python scripts/cleanup_models_auto.py
```

## Key Export Scripts (Current Versions)

- `scripts/export_vampnet_transformer_v2.py` - Exports transformer with ONNX-friendly custom ops
- `scripts/export_c2f_transformer_v2.py` - Exports C2F (Coarse-to-Fine) transformer
- `scripts/export_vampnet_codec_proper.py` - Exports codec encoder/decoder
- `scripts/transfer_weights_vampnet_v2.py` - Transfers weights with VampNet naming conventions
- `scripts/complete_weight_transfer.py` - Completes weight transfer including embeddings
- `scripts/vampnet_full_pipeline_fixed.py` - Current working pipeline demo

Note: Many scripts in `scripts/` are superseded versions. Look for "v2", "fixed", or "proper" suffixes for current versions.

## Current Limitations

1. **Fixed Sequence Length**: Transformer requires exactly 100 tokens (cannot handle variable lengths)
2. **Simplified Codec**: Uses LAC instead of full DAC codec
3. **No Dynamic Sampling**: Temperature/top-k sampling not yet implemented in ONNX

## Testing Models

```python
# Test exported models
python scripts/test_complete_model.py

# Run audio generation pipeline
python scripts/vampnet_full_pipeline_demo.py input.wav output.wav

# Benchmark performance
python examples/benchmark_performance.py
```

## Important Notes

- **Weight transfer is now complete** - Recent updates achieved full weight transfer from pretrained VampNet models
- **Use V2 exports** - Scripts with "v2" suffix use ONNX-friendly architectures that support weight transfer
- **Active model paths**:
  - Coarse: `onnx_models_fixed/coarse_transformer_v2_weighted.onnx`
  - C2F: `onnx_models_fixed/c2f_transformer_v2_weighted.onnx`
  - Encoder/Decoder: `models/vampnet_codec_*.onnx`
- VampNet uses GatedGELU activation and FFN dimension of 2560 (not standard GELU/5120)
- Custom ops require ONNX Runtime 1.15+ for proper execution
- Quantization can reduce model size by 46-55% with minimal quality loss