# VampNet Model Structure Analysis

## Overview
This document summarizes the key findings from analyzing the actual VampNet model structure, including both coarse and c2f (coarse-to-fine) models.

## Key Naming Patterns in VampNet

### 1. **Embeddings**
- `embedding.special.MASK`: Special mask token embedding
  - Coarse: `torch.Size([4, 8])` - 4 codebooks, 8 dimensions
  - C2F: `torch.Size([14, 8])` - 14 codebooks, 8 dimensions
- `embedding.out_proj.weight`: Output projection after embedding
  - Coarse: `torch.Size([1280, 32, 1])` 
  - C2F: `torch.Size([1280, 112, 1])`
- `embedding.out_proj.bias`: `torch.Size([1280])`

### 2. **Transformer Layers**
Pattern: `transformer.layers.{layer_idx}.{component}`

#### Normalization layers:
- `transformer.layers.{i}.norm_1.weight` - Pre-attention norm
- `transformer.layers.{i}.norm_3.weight` - Pre-FFN norm
- `transformer.norm.weight` - Final layer norm

#### Attention components:
- `transformer.layers.{i}.self_attn.w_qs.weight` - Query projection
- `transformer.layers.{i}.self_attn.w_ks.weight` - Key projection  
- `transformer.layers.{i}.self_attn.w_vs.weight` - Value projection
- `transformer.layers.{i}.self_attn.fc.weight` - Output projection
- `transformer.layers.{i}.self_attn.relative_attention_bias.weight` - Relative position bias (only in layer 0)

#### LoRA adapters (for all attention weights):
- `transformer.layers.{i}.self_attn.{w_qs/w_ks/w_vs/fc}.lora_A`
- `transformer.layers.{i}.self_attn.{w_qs/w_ks/w_vs/fc}.lora_B`

#### Feed-forward network:
- `transformer.layers.{i}.feed_forward.w_1.weight` - First linear layer
  - Shape: `torch.Size([5120, 1280])` - Projects from 1280 to 5120
- `transformer.layers.{i}.feed_forward.w_2.weight` - Second linear layer  
  - Shape: `torch.Size([1280, 2560])` - Projects from 2560 to 1280
- LoRA adapters for both layers

### 3. **Classifier/Output**
- `classifier.layers.0.weight_v` - Weight value (weight normalization)
  - Coarse: `torch.Size([4096, 1280, 1])` - 4 codebooks × 1024 vocab
  - C2F: `torch.Size([10240, 1280, 1])` - 10 codebooks × 1024 vocab
- `classifier.layers.0.weight_g` - Weight magnitude (weight normalization)
- `classifier.layers.0.bias` - Bias term

## Key Differences: VampNet vs Our ONNX Model

### 1. **Attention Implementation**
- **VampNet**: Uses separate `w_qs`, `w_ks`, `w_vs` linear layers
- **Our ONNX**: Uses combined `in_proj_weight` and `in_proj_bias`
- **Mapping needed**: Concatenate Q, K, V weights into single tensor

### 2. **FFN Architecture**
- **VampNet**: Uses GatedGELU activation with shape changes:
  - w_1: [5120, 1280] - expands to 5120
  - w_2: [1280, 2560] - reduces from 2560 
- **Our ONNX**: Standard FFN with 4x expansion
- **Key insight**: VampNet uses gated activation, splitting w_1 output

### 3. **Normalization**
- **VampNet**: Uses `norm_1` and `norm_3` naming
- **Our ONNX**: Uses `norm1` and `norm2` naming
- Both use RMSNorm

### 4. **Embedding Structure**
- **VampNet**: Complex CodebookEmbedding with special tokens
- **Our ONNX**: Simple embedding lookup tables
- Need to handle the special MASK token and output projection

### 5. **Output Layer**
- **VampNet**: Uses weight-normalized Conv1d as classifier
- **Our ONNX**: Uses simple Linear layers
- Need to compute actual weights from weight_v and weight_g

## Model Specifications

### Coarse Model
- **Layers**: 20 transformer layers
- **Hidden size**: 1280
- **Attention heads**: 20 (implied from 1280/64)
- **FFN dimensions**: 5120 (intermediate), 2560 (gated)
- **Codebooks**: 4
- **Vocab size**: 1024
- **Total parameters**: 335,893,664

### C2F Model
- **Layers**: 16 transformer layers
- **Hidden size**: 1280
- **Attention heads**: 20
- **FFN dimensions**: Same as coarse
- **Codebooks**: 10 (additional fine codebooks)
- **Vocab size**: 1024
- **Total parameters**: 295,437,440

## Weight Transfer Strategy

1. **Embeddings**:
   - Extract embedding weights from VampNet's complex structure
   - Handle the MASK token specially
   - Apply output projection if needed

2. **Attention Weights**:
   ```python
   # Concatenate VampNet's separate QKV into our combined format
   in_proj_weight = torch.cat([w_qs.weight, w_ks.weight, w_vs.weight], dim=0)
   ```

3. **FFN Weights**:
   - Account for GatedGELU architecture
   - w_1 produces both gate and value tensors
   - w_2 takes only the gated output

4. **Classifier**:
   ```python
   # Reconstruct weights from weight normalization
   weight = weight_v * (weight_g / torch.norm(weight_v, dim=1, keepdim=True))
   ```

5. **Handle LoRA weights**:
   - Can be merged into base weights or kept separate
   - For base model, merge: `weight = weight + lora_B @ lora_A`

## Implementation Notes

- VampNet includes LoRA adapters for fine-tuning
- Uses FiLM (Feature-wise Linear Modulation) layers
- Implements relative position bias only in first layer
- Uses dropout throughout (not needed for inference)
- Weight normalization in classifier needs special handling