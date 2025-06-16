# VampNet ONNX Troubleshooting Guide

## Common Issues and Solutions

### 1. Poor Audio Quality from ONNX

**Symptoms**: ONNX generates noise or "rubbish" audio while VampNet works fine

**Possible Causes**:
- Iterative generation not working correctly
- Mask token handling issues  
- Latent conversion problems
- Sampling strategy differences

**Debugging Steps**:
1. Check intermediate outputs:
   ```python
   # After each iteration, check:
   print(f"Iteration {i}: min={z.min()}, max={z.max()}, masked={(z==1024).sum()}")
   ```

2. Verify latent conversion:
   ```python
   # Ensure latents match between VampNet and ONNX
   latents_vampnet = vampnet.coarse.embedding.from_codes(z, codec)
   latents_onnx = onnx_generator.codes_to_latents(z)
   print(f"Latent diff: {(latents_vampnet - latents_onnx).abs().max()}")
   ```

3. Check sampling:
   ```python
   # Ensure temperature and sampling match
   # VampNet uses typical_filtering=True by default
   ```

### 2. C2F Model NaN Issues

**Symptoms**: C2F outputs NaN or Inf values

**Status**: Known issue, occurs in both VampNet and ONNX for certain seeds

**Workaround**:
```python
try:
    c2f_output = c2f_model(latents)
    if torch.isnan(c2f_output).any():
        print("C2F produced NaN, using coarse only")
        # Fall back to coarse-only generation
except:
    # Use coarse output only
```

### 3. Shape Mismatches

**Common Shapes**:
- Codes: `[batch, n_codebooks, seq_len]`
- Latents: `[batch, n_codebooks * latent_dim, seq_len]`
- Logits (coarse): `[batch, vocab_size, seq_len]`
- Logits (C2F): `[batch, vocab_size, seq_len * n_predict_codebooks]`

**Debug**:
```python
print(f"Codes: {codes.shape}")  # Should be [1, 14, 100]
print(f"Latents: {latents.shape}")  # Should be [1, 112, 100]
print(f"Logits: {logits.shape}")  # Coarse: [1, 1025, 100]
```

### 4. Mask Issues

**VampNet Mask Semantics**:
- 1 = masked (to be generated)
- 0 = unmasked (keep original)

**Common Mistakes**:
- Wrong mask polarity
- Not masking the right codebooks
- Mask shape doesn't match codes

**Correct Mask Creation**:
```python
mask = vampnet.build_mask(z, signal, upper_codebook_mask=3)
# Or manually:
mask = torch.ones_like(z)
mask[:, 3:, :] = 0  # Don't mask codebooks 3+
```

### 5. Wrong Model Versions

**Check Model Paths**:
```python
# Correct models:
encoder: "scripts/models/vampnet_encoder_prepadded.onnx"
coarse: "vampnet_transformer_v11.onnx"
decoder: "scripts/models/vampnet_codec_decoder.onnx"
c2f: "vampnet_c2f_transformer_v15.onnx"

# Old/wrong models:
# Anything with v1-v10 (missing attention layers)
# Anything without "prepadded" (wrong encoder)
```

### 6. Token Sequence Length

**Issue**: ONNX models fixed at 100 tokens

**Calculate Tokens**:
```python
hop_length = 768
sample_rate = 44100
n_tokens = audio_samples // hop_length
# For 100 tokens: need exactly 76,800 samples
```

### 7. Debugging Integration

**Step-by-Step Verification**:
```python
# 1. Check encoding matches
z_vampnet = vampnet.encode(signal)
z_onnx = onnx_encode(signal)
print(f"Encoding match: {(z_vampnet == z_onnx).float().mean():.1%}")

# 2. Check single forward pass
with torch.no_grad():
    # Get logits from both
    logits_vampnet = vampnet.coarse(latents)
    logits_onnx = onnx_model(latents)
    
# 3. Check sampling
tokens_vampnet = sample_from_logits(logits_vampnet, temp=1.0)
tokens_onnx = sample_from_logits(logits_onnx, temp=1.0)
```

## Quick Diagnostic Script

```python
def diagnose_onnx_issue(vampnet, onnx_generator, audio_signal):
    """Quick diagnostic to identify where ONNX diverges from VampNet"""
    
    # 1. Encode
    z_vamp = vampnet.encode(audio_signal)
    z_onnx = onnx_encode(audio_signal)
    print(f"1. Encoding match: {(z_vamp == z_onnx).float().mean():.1%}")
    
    # 2. Mask
    mask = vampnet.build_mask(z_vamp, audio_signal)
    print(f"2. Mask coverage: {mask.float().mean():.1%}")
    
    # 3. First iteration only
    z_masked = z_vamp.clone()
    z_masked[mask.bool()] = 1024
    
    # 4. Latents
    latents_vamp = vampnet.coarse.embedding.from_codes(z_masked, vampnet.codec)
    latents_onnx = onnx_generator.codes_to_latents(z_masked)
    print(f"3. Latent match: {torch.allclose(latents_vamp, latents_onnx, atol=1e-5)}")
    
    # 4. Forward pass
    logits_vamp = vampnet.coarse(latents_vamp)
    logits_onnx = onnx_generator.forward(z_masked)
    print(f"4. Logits correlation: {torch.corrcoef(torch.stack([logits_vamp.flatten(), logits_onnx.flatten()]))[0,1]:.4f}")
    
    # 5. Sample
    from scripts.iterative_generation import sample_from_logits
    tokens_vamp = sample_from_logits(logits_vamp, temperature=1.0)
    tokens_onnx = sample_from_logits(logits_onnx, temperature=1.0)
    print(f"5. Token match rate: {(tokens_vamp == tokens_onnx).float().mean():.1%}")
```

## When All Else Fails

1. **Use Pipeline Comparison Scripts**: The scripts in `pipeline_comparison/` are known to work
2. **Start Fresh**: Sometimes it's easier to start over than debug
3. **Check Git History**: Look for commits where things worked
4. **Simplify**: Remove C2F, remove iterations, test core functionality

Remember: The individual components work. The issue is in integration or subtle differences in iterative generation.