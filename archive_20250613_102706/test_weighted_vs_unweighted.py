#!/usr/bin/env python3
"""
Test weighted vs unweighted ONNX models to see if weight transfer helped.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_models():
    """Compare weighted and unweighted model outputs."""
    
    print("=== Testing Weighted vs Unweighted Models ===\n")
    
    # Models to test
    models = {
        "Unweighted": "scripts/onnx_models_fixed/coarse_transformer_v2.onnx",
        "Weighted": "scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx",
    }
    
    # Create test input
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    
    # Input codes with a pattern
    codes = np.zeros((batch_size, n_codebooks, seq_len), dtype=np.int64)
    for i in range(n_codebooks):
        codes[:, i, :] = (np.arange(seq_len) + i * 100) % 1024
    
    # Mask middle portion
    mask = np.zeros((batch_size, n_codebooks, seq_len), dtype=bool)
    mask[:, :, 40:60] = True
    
    print(f"Test input:")
    print(f"  Codes shape: {codes.shape}")
    print(f"  Masked positions: {mask.sum()}")
    print(f"  Sample codes: {codes[0, 0, :10]}")
    
    results = {}
    
    for name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"\n{name} model not found at {model_path}")
            continue
            
        print(f"\n{name} Model:")
        session = ort.InferenceSession(model_path)
        
        # Run inference
        output = session.run(None, {
            'codes': codes,
            'mask': mask
        })[0]
        
        results[name] = output
        
        # Analyze output
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min()}, {output.max()}]")
        print(f"  Unique values: {len(np.unique(output))}")
        print(f"  Sample output: {output[0, 0, 40:50]}")
        
        # Check if positions were changed
        changed = np.sum(codes != output)
        masked_changed = np.sum((codes != output) & mask)
        print(f"  Total changed: {changed}")
        print(f"  Masked positions changed: {masked_changed}/{mask.sum()}")
        
        # Check for mask tokens
        mask_tokens = np.sum(output == 1024)
        print(f"  Mask tokens (1024) in output: {mask_tokens}")
    
    # Compare outputs
    if len(results) == 2:
        print("\n=== Comparison ===")
        unweighted = results["Unweighted"]
        weighted = results["Weighted"]
        
        # Compare diversity
        print(f"\nToken diversity:")
        print(f"  Unweighted: {len(np.unique(unweighted))} unique tokens")
        print(f"  Weighted: {len(np.unique(weighted))} unique tokens")
        
        # Compare patterns
        diff = np.sum(unweighted != weighted)
        total = unweighted.size
        print(f"\nDifference between outputs: {diff}/{total} ({100*diff/total:.1f}%)")
        
        # Check if weighted model produces more reasonable outputs
        print(f"\nOutput statistics:")
        print(f"  Unweighted mean: {unweighted.mean():.1f}, std: {unweighted.std():.1f}")
        print(f"  Weighted mean: {weighted.mean():.1f}, std: {weighted.std():.1f}")
        
        # Visual comparison of a slice
        print(f"\nSample outputs (codebook 0, positions 40-50):")
        print(f"  Original:   {codes[0, 0, 40:50]}")
        print(f"  Unweighted: {unweighted[0, 0, 40:50]}")
        print(f"  Weighted:   {weighted[0, 0, 40:50]}")
        
    print("\n=== Summary ===")
    print("If the weighted model produces:")
    print("- Similar outputs to unweighted -> weights didn't transfer properly")
    print("- More diverse tokens -> partial weight transfer")
    print("- Very different patterns -> successful weight transfer")


if __name__ == "__main__":
    test_models()