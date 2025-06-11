# ONNX Generation Mismatch Analysis

## Executive Summary

Despite achieving 97.9% token match between VampNet and ONNX encoders, the ONNX transformers produce significantly different audio outputs than VampNet. This analysis identifies the root causes and provides recommendations.

## Key Findings

### 1. Encoder Success ✅
- The pre-padded ONNX encoder produces tokens that are 97.9% identical to VampNet's DAC encoder
- This was achieved by:
  - Using VampNet's actual codec weights
  - Pre-padding audio to multiples of 768 samples (hop length)
  - Exporting with fixed input sizes to avoid ONNX tracing issues

### 2. Transformer Issues ❌

#### Incomplete Weight Transfer
The ONNX transformer models have **only partial weight transfer** from VampNet:

```
Transferred: 121 transformer layer parameters ✅
Missing:     Embeddings (using random initialization) ❌
Missing:     Output classifiers (shape mismatch) ❌
```

Weight statistics comparison:
- **ONNX embeddings**: mean≈0.0000, std=0.0200 (random initialization)
- **VampNet embeddings**: mean=-0.0190, std=0.3573 (trained weights)

#### Architectural Differences

1. **Embedding Approach**
   - VampNet: Uses codec embeddings directly (no explicit embedding tables)
   - ONNX: Uses standard embedding tables (1025 × 1280 per codebook)

2. **Output Dimensions**
   - VampNet: 4096 output classes
   - ONNX: 1025 output classes (1024 vocab + 1 mask token)

3. **Special Token Handling**
   - VampNet: Has special mask embeddings (shape: [4, 8])
   - ONNX: Standard mask token (index 1024)

### 3. Generation Quality Impact

Testing weighted vs unweighted ONNX models shows:

```
Unweighted model: 322 unique tokens, repetitive patterns
Weighted model:   357 unique tokens, more variation
```

While the weighted model performs better, both produce significantly different outputs than VampNet due to missing components.

## Root Causes

### 1. Fundamental Architecture Mismatch
VampNet's architecture is tightly coupled with its codec:
- Embeddings come from the codec, not learnable tables
- The transformer expects codec-specific representations
- Output projections are designed for codec vocabulary

### 2. Incomplete Export Process
The current export process:
- ✅ Successfully exports transformer layer weights
- ❌ Cannot export codec-based embeddings (different paradigm)
- ❌ Cannot handle output dimension mismatch (4096 vs 1025)

### 3. ONNX Limitations
- Fixed sequence lengths due to tracing
- Difficulty representing codec-based embeddings
- No native support for VampNet's special token handling

## Impact on Audio Generation

The incomplete weight transfer results in:
1. **Different Token Distributions**: ONNX generates tokens with different statistical properties
2. **Reduced Quality**: Generated audio lacks the coherence of VampNet's output
3. **Inconsistent Masking**: Different handling of mask tokens affects generation

## Recommendations

### Short-term Solutions

1. **Complete Weight Transfer**
   - Initialize ONNX embeddings from codec outputs
   - Adjust output dimensions to match VampNet (4096)
   - Transfer all available weights, not just transformer layers

2. **Improve Architecture Alignment**
   - Modify ONNX model to better match VampNet's approach
   - Implement codec-based embedding lookup
   - Match exact activation functions and normalization

### Long-term Solutions

1. **Alternative Export Methods**
   - Consider TorchScript for preserving exact architecture
   - Explore custom ONNX operators for codec integration
   - Use torch.jit.trace with representative inputs

2. **Hybrid Approach**
   - Keep codec in PyTorch, export only transformer
   - Use ONNX for transformer inference only
   - Maintain VampNet's embedding/output logic

3. **Training New Components**
   - Train new embedding layers while keeping transformer weights
   - Fine-tune output classifiers for ONNX vocabulary
   - Validate against VampNet outputs

## Technical Details

### Weight Transfer Status

| Component | Status | Issue |
|-----------|---------|--------|
| Transformer Layers | ✅ Partial | Only 121/374 parameters |
| Embeddings | ❌ | Random initialization |
| Positional Encoding | ⚠️ | Sinusoidal instead of learned |
| Output Classifiers | ❌ | Shape mismatch (4096 vs 1025) |
| Normalization | ✅ | RMSNorm transferred |
| Attention | ✅ | Weights transferred |
| FFN | ✅ | GatedGELU weights transferred |

### File Structure

```
Current ONNX Models:
- scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx (1.38 GB)
- scripts/onnx_models_fixed/c2f_transformer_v2_weighted.onnx (1.22 GB)
- scripts/models/vampnet_encoder_prepadded.onnx (91.4 MB)
- scripts/models/vampnet_codec_decoder.onnx (90.8 MB)
```

## Conclusion

The ONNX models successfully replicate VampNet's encoder behavior but fail to match the transformer generation quality due to fundamental architectural differences and incomplete weight transfer. The core issue is that VampNet's tight integration with its codec cannot be easily separated into standard ONNX components.

To achieve matching quality, either:
1. Modify the export process to preserve VampNet's exact architecture
2. Accept the quality difference and optimize the ONNX models independently
3. Use a hybrid approach keeping critical components in PyTorch

## Next Steps

1. **Immediate**: Document the current limitations for users
2. **Short-term**: Attempt complete weight transfer with architecture modifications
3. **Long-term**: Explore alternative export strategies or hybrid deployments