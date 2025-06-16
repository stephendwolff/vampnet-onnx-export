# VampNet Mask Generation Fix Summary

## The Issue

When comparing VampNet and ONNX mask generation with parameters:
- `rand_mask_intensity=0.0`
- `periodic_prompt=7`
- `upper_codebook_mask=3`

VampNet produced mostly zeros (unmasked) while ONNX produced mostly ones (masked).

## Root Cause

The issue was in the `codebook_mask` implementation. The ONNX version had inverted logic:

**VampNet (correct):**
```python
def codebook_mask(mask, val1):
    mask[:, val1:, :] = 1  # Masks codebooks >= val1
```

**ONNX (incorrect):**
```python
def _codebook_mask(shape, upper_codebook_mask):
    mask[:, :upper_codebook_mask, :] = True  # Was masking codebooks < upper_codebook_mask
```

## Key Insights

1. **Mask Semantics**: In VampNet, `1` = MASKED (will be replaced), `0` = UNMASKED (will be preserved)

2. **Mask Building Process**:
   - Start with `linear_random(rand_mask_intensity)` - returns probability-based mask
   - Apply `periodic_mask` using AND operation - preserves periodic positions as 0s
   - Apply `codebook_mask` - forces codebooks >= threshold to 1 (masked)

3. **With `rand_mask_intensity=0.0`**:
   - `linear_random(0.0)` returns all zeros (nothing masked)
   - `mask_and(zeros, anything)` = zeros
   - `codebook_mask` then sets upper codebooks to 1

## The Fix

Changed line 73 in `vampnet_onnx/mask_generator_proper.py`:
```python
# Before (incorrect):
mask[:, :upper_codebook_mask, :] = True

# After (correct):
mask[:, upper_codebook_mask:, :] = True
```

This ensures codebooks >= `upper_codebook_mask` are masked, matching VampNet's behavior.

## Verification

After the fix:
- Deterministic tests show 100% match between VampNet and ONNX
- Both produce 78.6% mask ratio with the test parameters
- Lower codebooks (0-2) remain unmasked (zeros)
- Upper codebooks (3-13) are fully masked (ones)

## Impact

This fix ensures that ONNX mask generation exactly matches VampNet's behavior, which is critical for:
- Correct token generation during inference
- Maintaining the intended masking strategy
- Ensuring lower codebooks (coarse tokens) are preserved while upper codebooks (fine details) are regenerated