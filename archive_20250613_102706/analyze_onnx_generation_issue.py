#!/usr/bin/env python3
"""
Analyze why ONNX models produce different outputs than VampNet.
Focus on the mask token handling and generation strategy.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def analyze_onnx_generation():
    """Analyze ONNX generation behavior."""
    
    print("=== Analyzing ONNX Generation Issues ===\n")
    
    # Load ONNX coarse model
    coarse_path = Path("scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx")
    if not coarse_path.exists():
        print(f"Coarse model not found at {coarse_path}")
        return
        
    session = ort.InferenceSession(str(coarse_path))
    
    # Test 1: Check if model preserves unmasked tokens
    print("1. Testing Token Preservation")
    
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    
    # Create input with specific pattern
    codes = np.zeros((batch_size, n_codebooks, seq_len), dtype=np.int64)
    for i in range(n_codebooks):
        codes[:, i, :] = np.arange(seq_len) + i * 100
    
    # No masking - model should preserve all tokens
    mask = np.zeros_like(codes, dtype=bool)
    
    output_no_mask = session.run(None, {'codes': codes, 'mask': mask})[0]
    
    preserved = np.all(codes == output_no_mask)
    print(f"  All tokens preserved (no mask): {preserved}")
    if not preserved:
        diff_count = np.sum(codes != output_no_mask)
        print(f"  Changed tokens: {diff_count}")
    
    # Test 2: Analyze masked generation
    print("\n2. Testing Masked Generation")
    
    # Mask half the sequence
    mask = np.zeros_like(codes, dtype=bool)
    mask[:, :, 50:] = True
    
    output_masked = session.run(None, {'codes': codes, 'mask': mask})[0]
    
    # Check unmasked positions
    unmasked_preserved = np.all(codes[:, :, :50] == output_masked[:, :, :50])
    print(f"  Unmasked tokens preserved: {unmasked_preserved}")
    
    # Check masked positions
    masked_changed = np.sum(codes[:, :, 50:] != output_masked[:, :, 50:])
    total_masked = n_codebooks * 50
    print(f"  Masked tokens changed: {masked_changed}/{total_masked}")
    
    # Check for mask tokens (1024)
    mask_tokens = np.sum(output_masked == 1024)
    print(f"  Mask tokens (1024) in output: {mask_tokens}")
    
    # Test 3: Analyze token distribution
    print("\n3. Token Distribution Analysis")
    
    unique_input = len(np.unique(codes))
    unique_output = len(np.unique(output_masked))
    print(f"  Input unique tokens: {unique_input}")
    print(f"  Output unique tokens: {unique_output}")
    print(f"  Output range: [{output_masked.min()}, {output_masked.max()}]")
    
    # Test 4: Check if model can handle mask tokens in input
    print("\n4. Testing Mask Token Handling")
    
    # Put some mask tokens (1024) in the input
    codes_with_mask = codes.copy()
    codes_with_mask[:, :, 25:35] = 1024
    
    # Only mask positions 50+
    output_with_mask_input = session.run(None, {
        'codes': codes_with_mask, 
        'mask': mask
    })[0]
    
    # Check if mask tokens in unmasked positions are preserved
    mask_preserved = np.all(codes_with_mask[:, :, 25:35] == output_with_mask_input[:, :, 25:35])
    print(f"  Mask tokens in unmasked positions preserved: {mask_preserved}")
    
    # Test 5: Iterative generation (like VampNet)
    print("\n5. Testing Iterative Generation Strategy")
    
    # Start with all mask tokens at masked positions
    codes_iter = codes.copy()
    codes_iter[mask] = 1024
    
    # First iteration
    output_iter1 = session.run(None, {'codes': codes_iter, 'mask': mask})[0]
    
    # Replace only masked positions
    codes_iter[mask] = output_iter1[mask]
    
    # Check if we still have mask tokens
    mask_tokens_iter1 = np.sum(codes_iter == 1024)
    print(f"  After iteration 1: mask tokens = {mask_tokens_iter1}")
    
    # The issue: ONNX model might be outputting mask tokens
    # which should be replaced with valid tokens
    
    # Test 6: Fix mask tokens in output
    print("\n6. Testing Mask Token Fix")
    
    # Clip to valid range
    output_fixed = np.clip(output_masked, 0, 1023)
    fixed_count = np.sum(output_masked != output_fixed)
    print(f"  Tokens fixed by clipping: {fixed_count}")
    
    # Alternative: Replace mask tokens with a valid token
    output_replaced = output_masked.copy()
    output_replaced[output_replaced == 1024] = 0  # or sample from distribution
    replaced_count = np.sum(output_masked == 1024)
    print(f"  Tokens fixed by replacement: {replaced_count}")
    
    # Summary
    print("\n=== Summary ===")
    print("Key findings:")
    print("1. ONNX model outputs mask tokens (1024) which are invalid for decoder")
    print("2. These need to be clipped or replaced before decoding")
    print("3. The model may not be properly trained to avoid outputting mask tokens")
    print("4. VampNet likely has additional logic to handle this")
    
    return session, codes, output_masked


if __name__ == "__main__":
    session, codes, output = analyze_onnx_generation()