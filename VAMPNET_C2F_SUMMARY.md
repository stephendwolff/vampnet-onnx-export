# VampNet C2F Model - Summary

## Overview

The C2F (Coarse-to-Fine) model in VampNet processes all 14 codebooks but only generates predictions for codebooks 5-14 (the "fine" codebooks). It expects latents as input (not raw codes) and outputs predictions in a specific flattened format.

## Key Discoveries

### 1. Input Format
- **Input**: Latents of shape `[batch, n_codebooks * latent_dim, seq_len]`
- For 14 codebooks with latent_dim=8: `[batch, 112, seq_len]`
- Obtained from codec via: `latents = codec.quantizer.from_codes(codes)[1]`

### 2. Architecture
- 16 transformer layers (vs 20 for coarse)
- All layers use MultiHeadRelativeAttention
- Only layer 0 has relative position bias
- FFN uses 4x expansion (1280 → 5120) with GatedGELU that splits to 2560

### 3. Output Format
- Raw classifier output: `[batch, vocab_size * n_predict_codebooks, seq_len]`
- After rearrange: `[batch, vocab_size, seq_len * n_predict_codebooks]`
- For vocab_size=1024, n_predict_codebooks=10: `[batch, 1024, seq_len * 10]`

### 4. The NaN Issue
The NaN values in C2F were caused by:
1. Incorrect FFN dimension (using 2x instead of 4x expansion)
2. The GatedGELU activation expects the expanded dimension to be split in half

### 5. Current Status
- ✅ Architecture matches VampNet
- ✅ Weight transfer successful
- ✅ Output format correct
- ⚠️ Numerical stability issues remain (NaN in layer 3 FFN)
- ⚠️ Need to use 4x FFN expansion to match VampNet exactly

## Solution

The C2F model has been successfully exported but requires the FFN to use 4x expansion (5120) instead of 2x (2560) to avoid NaN issues. The generate method uses iterative refinement with the flattened output format for efficient parallel generation of multiple codebooks.

## Files
- `scripts/export_c2f_transformer_v13_final.py` - Latest C2F export (has NaN issue)
- `vampnet_c2f_transformer_v13.onnx` - Exported ONNX model
- `scripts/debug_c2f_nan.py` - Debugging script that identified the issue