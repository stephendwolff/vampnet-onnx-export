# Weight Transfer Technical Notes

## Overview

This document provides technical details about the weight transfer process from VampNet to ONNX models, including specific challenges and solutions.

## Weight Transfer Pipeline

### Current Process

1. **Export Model Structure** (`export_vampnet_transformer_v2.py`)
   - Creates ONNX-compatible transformer architecture
   - Uses custom operators for ONNX compatibility
   - Exports with random weight initialization

2. **Transfer Weights** (`transfer_weights_vampnet_v2.py`)
   - Loads VampNet checkpoint
   - Maps VampNet parameter names to ONNX names
   - Transfers compatible weights

3. **Complete Transfer** (`complete_weight_transfer.py`)
   - Attempts to transfer embeddings and classifiers
   - Handles special tokens and projections
   - Exports final ONNX model

### Weight Mapping Details

#### Successfully Transferred Components

```python
# Transformer layer mapping
VampNet: transformer.layers.{i}.self_attn.w_qs.weight
ONNX:    layers.{i}.self_attn.w_q.weight

VampNet: transformer.layers.{i}.self_attn.fc.weight  
ONNX:    layers.{i}.self_attn.w_o.weight

VampNet: transformer.layers.{i}.norm_1.weight
ONNX:    layers.{i}.norm1.weight

VampNet: transformer.layers.{i}.ff.w_1.weight (5120, 1280)
ONNX:    layers.{i}.ffn.w_1.weight (5120, 1280)
```

#### Failed Transfers

```python
# Embeddings
VampNet: Uses codec directly (no embedding tables)
ONNX:    embedding.embeddings[i].weight (1025, 1280)
Result:  Random initialization

# Output classifiers  
VampNet: classifier.layers.{i}.weight_v (4096, 1280, 1)
ONNX:    output_projs[i].weight (1025, 1280)
Result:  Shape mismatch, no transfer

# Special tokens
VampNet: embedding.special.MASK (4, 8)
ONNX:    No equivalent structure
Result:  Not transferred
```

## Architecture Comparison

### VampNet Architecture

```
Input: Audio → DAC Codec → Tokens
      ↓
Embedding: Tokens → Codec Embeddings → Linear Projection
      ↓
Transformer: 20 layers with:
  - Multi-head Attention (20 heads)
  - RMSNorm
  - GatedGELU FFN (1280 → 5120 → 2560 → 1280)
  - FiLM conditioning
      ↓
Output: Linear → 4096 classes per codebook
```

### ONNX Architecture

```
Input: Tokens (pre-encoded)
      ↓
Embedding: Learnable embeddings (1025 × 1280)
      ↓
Transformer: 20 layers with:
  - Multi-head Attention (20 heads)
  - RMSNorm  
  - GatedGELU FFN (1280 → 5120 → 2560 → 1280)
  - FiLM conditioning
      ↓
Output: Linear → 1025 classes per codebook
```

## Key Differences

### 1. Embedding Generation

**VampNet:**
```python
# Pseudocode
embeddings = codec.embed(tokens)  # Uses DAC codec
projected = embedding_projection(embeddings)
```

**ONNX:**
```python
# Standard embedding lookup
embeddings = embedding_table[tokens]
```

### 2. Vocabulary Size

- VampNet: 4096 (codec-specific)
- ONNX: 1024 + 1 mask token = 1025

### 3. Special Token Handling

**VampNet:**
- Per-codebook mask embeddings
- Shape: [n_codebooks, embedding_dim]

**ONNX:**
- Single mask token (index 1024)
- Same embedding space as regular tokens

## Weight Transfer Script Analysis

### `transfer_weights_vampnet_v2.py`

Successful mappings:
- Attention: Q, K, V, O projections ✅
- Layer norms: RMSNorm weights ✅
- FFN: Both linear layers ✅
- FiLM: Conditioning weights ✅

Failed mappings:
- Embeddings: No source weights ❌
- Classifiers: Shape mismatch ❌
- Positional encoding: Different approach ❌

### `complete_weight_transfer.py`

Attempts to handle missing components:

```python
# Embedding initialization (not from VampNet)
nn.init.normal_(emb.weight, mean=0.0, std=0.02)

# Positional encoding (sinusoidal, not learned)
pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
pos_encoding[0, :, 1::2] = torch.cos(position * div_term)

# Classifier mapping fails due to shape
# VampNet: (4096, 1280) 
# ONNX: (1025, 1280)
```

## Validation Results

### Token Generation Test

```
Input: Sequential tokens [0, 1, 2, ..., 99]
Mask: Positions 40-60

Unweighted ONNX: [109, 109, 109, ...] (repetitive)
Weighted ONNX: [337, 337, 308, 4, 1020, ...] (more varied)
VampNet: Not directly comparable due to architecture differences
```

### Weight Statistics

```
Component               | VampNet      | ONNX (weighted)
------------------------|--------------|----------------
Embedding mean          | -0.0190      | 0.0000
Embedding std           | 0.3573       | 0.0200
Layer norm mean         | 0.0425       | 0.0425 ✓
Attention weight std    | ~0.1         | ~0.1 ✓
```

## Recommendations for Complete Transfer

### Option 1: Exact Architecture Match

1. Implement codec embedding lookup in ONNX
2. Adjust vocabulary size to 4096
3. Add special mask token handling
4. Retrain output projections if needed

### Option 2: Adapter Approach

1. Keep current ONNX architecture
2. Add adapter layers to map between paradigms
3. Train adapters while freezing transformer weights

### Option 3: Hybrid Deployment

1. Use PyTorch for embeddings and output
2. Use ONNX only for transformer layers
3. Combine at inference time

## Code Snippets for Future Work

### Extracting Codec Embeddings

```python
# Get embeddings for all possible tokens
all_tokens = torch.arange(1024).reshape(1, 1, -1)
codec_embeddings = vampnet_model.encode(all_tokens)
# Save as embedding initialization
torch.save(codec_embeddings, 'codec_embeddings.pth')
```

### Adjusting Output Dimensions

```python
# In export script
class VampNetTransformerV2(nn.Module):
    def __init__(self, ...):
        # Change from 1025 to 4096
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, 4096)  # Match VampNet
            for _ in range(n_codebooks)
        ])
```

### Complete Weight Transfer

```python
# Proposed complete transfer
def transfer_all_weights(vampnet_model, onnx_model):
    # 1. Transfer transformer weights (current)
    transfer_transformer_weights(...)
    
    # 2. Initialize embeddings from codec
    init_embeddings_from_codec(...)
    
    # 3. Map classifier with dimension adjustment
    map_classifier_with_resize(...)
    
    # 4. Copy special tokens
    copy_special_tokens(...)
```

## Conclusion

The weight transfer is currently incomplete due to fundamental architectural differences between VampNet and the ONNX export. While transformer weights transfer successfully, the critical embedding and output components require architectural changes or alternative approaches to achieve comparable generation quality.