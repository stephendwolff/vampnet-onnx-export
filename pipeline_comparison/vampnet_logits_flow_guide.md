# VampNet Token-to-Logits Flow Guide

## Overview

This guide explains the correct way VampNet processes tokens to generate logits, based on the analysis of the codebase. Understanding this flow is crucial for proper ONNX export and inference.

## Key Components

### 1. Token Input Format
- **Shape**: `[batch_size, n_codebooks, seq_len]`
- **Values**: 0-1023 for regular tokens, 1024 for mask token
- **Codebooks**: Coarse model uses 4 codebooks, C2F uses 14 (4 coarse + 10 fine)

### 2. Embedding Architecture

VampNet uses a unique embedding architecture that differs from standard transformers:

```python
# Key parameters
vocab_size = 1024  # Regular tokens
mask_token = 1024  # Special mask token
latent_dim = 8     # Embedding dimension per codebook (NOT d_model!)
d_model = 1280     # Model dimension (coarse) or 768 (C2F)
```

#### The Embedding Process:

1. **Codebook Embeddings**: Each codebook has its own embedding table of size `(vocab_size + 1) x latent_dim`
   - Regular tokens (0-1023) map to embeddings
   - Mask token (1024) has a special embedding at the last position

2. **Concatenation**: Embeddings from all codebooks are concatenated
   - Results in shape: `[batch, seq_len, n_codebooks * latent_dim]`

3. **Projection**: A Conv1d layer projects from `n_codebooks * latent_dim` to `d_model`
   - Uses kernel_size=1 (equivalent to linear projection)
   - Final output: `[batch, seq_len, d_model]`

### 3. The Complete Forward Flow

```python
# Step 1: Apply mask to input tokens
masked_codes = codes.clone()
masked_codes[mask] = mask_token  # Replace masked positions with 1024

# Step 2: Get embeddings (the critical step!)
# This is what embedding.from_codes() does:
embeddings = []
for i in range(n_codebooks):
    cb_codes = masked_codes[:, i, :]  # [batch, seq_len]
    cb_emb = embedding_table[i](cb_codes)  # [batch, seq_len, latent_dim]
    embeddings.append(cb_emb)

# Concatenate: [batch, seq_len, n_codebooks * latent_dim]
embeddings = torch.cat(embeddings, dim=-1)

# Project to model dimension: [batch, seq_len, d_model]
x = conv1d_projection(embeddings)

# Step 3: Add positional encoding
x = x + positional_encoding[:, :seq_len, :]

# Step 4: Pass through transformer layers
for layer in transformer_layers:
    # Standard transformer operations
    x = layer(x)

# Step 5: Final normalization
x = final_norm(x)

# Step 6: Generate logits for each codebook
logits = []
for i in range(n_output_codebooks):
    cb_logits = output_projection[i](x)  # [batch, seq_len, vocab_size]
    logits.append(cb_logits)

# Stack: [batch, n_codebooks, seq_len, vocab_size]
logits = torch.stack(logits, dim=1)
```

## Important Details

### 1. Mask Token Handling
- The mask token (1024) is NOT out-of-vocabulary
- It has dedicated embeddings in each codebook's embedding table
- Must be applied BEFORE embedding lookup

### 2. Embedding Dimensions
- **Critical**: Each codebook uses `latent_dim=8`, not `d_model`!
- Total concatenated dimension: `n_codebooks * 8`
- For coarse model: `4 * 8 = 32` → projected to 1280
- For C2F model: `10 * 8 = 80` → projected to 768

### 3. Output Logits
- Shape: `[batch, n_codebooks, seq_len, vocab_size]`
- VampNet outputs `vocab_size=1024` classes
- ONNX models often output `vocab_size+1=1025` to include mask token

### 4. Conditioning vs Generation Codebooks
- Coarse model: All 4 codebooks are generated
- C2F model: First 4 codebooks are conditioning (from coarse), last 10 are generated
- Only generation codebooks get masked and have output projections

## Common Mistakes to Avoid

1. **Wrong Embedding Dimension**: Using `d_model` instead of `latent_dim=8` for codebook embeddings
2. **Missing Mask Token**: Not including the mask token in embedding tables
3. **Wrong Masking**: Applying mask after embedding instead of before
4. **Incorrect Concatenation**: Summing embeddings instead of concatenating
5. **Missing Projection**: Forgetting the Conv1d projection from concatenated embeddings to d_model

## Verification Tips

To verify correct implementation:

1. **Check Embedding Shapes**:
   - Codebook embeddings: `[vocab_size+1, 8]`
   - After concatenation: `[batch, seq_len, n_codebooks * 8]`
   - After projection: `[batch, seq_len, d_model]`

2. **Test Mask Token**:
   - Ensure mask token (1024) doesn't cause index errors
   - Verify it produces valid embeddings

3. **Compare Logits**:
   - Logits should be in reasonable range (typically -10 to 10)
   - Distribution should match VampNet's outputs
   - Correlation with original model should be > 0.9

## Example Code for Correct Implementation

```python
class CorrectVampNetEmbedding(nn.Module):
    def __init__(self, n_codebooks, vocab_size=1024, latent_dim=8, d_model=1280):
        super().__init__()
        # One embedding table per codebook, including mask token
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, latent_dim)  # +1 for mask token
            for _ in range(n_codebooks)
        ])
        # Projection from concatenated embeddings to model dimension
        self.projection = nn.Conv1d(n_codebooks * latent_dim, d_model, kernel_size=1)
    
    def forward(self, codes):
        # codes: [batch, n_codebooks, seq_len]
        batch, n_cb, seq_len = codes.shape
        
        # Get embeddings for each codebook
        embs = []
        for i in range(n_cb):
            emb = self.embeddings[i](codes[:, i, :])  # [batch, seq_len, latent_dim]
            embs.append(emb)
        
        # Concatenate and project
        x = torch.cat(embs, dim=-1)  # [batch, seq_len, n_cb * latent_dim]
        x = x.transpose(1, 2)  # [batch, n_cb * latent_dim, seq_len]
        x = self.projection(x)  # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        
        return x
```

## References

Key files that demonstrate the correct implementation:
- `scripts/export_coarse_with_logits.py` - Shows how to extract logits
- `scripts/codebook_embedding_correct_v2.py` - Correct embedding architecture
- `notebooks/compare_vampnet_onnx_logits.ipynb` - Logits comparison and validation