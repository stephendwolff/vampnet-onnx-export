#!/usr/bin/env python3
"""
Test transformer inference to debug generation issues.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_transformer_inference():
    """Test individual transformer models."""
    
    print("=== Testing Transformer Inference ===\n")
    
    # Test coarse transformer
    coarse_path = "scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx"
    
    if not Path(coarse_path).exists():
        print(f"Coarse transformer not found at {coarse_path}")
        return
        
    print("1. Testing Coarse Transformer")
    session = ort.InferenceSession(coarse_path)
    
    # Check inputs
    print("\nInputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
    
    # Check outputs  
    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")
    
    # Create test input
    batch_size = 1
    n_codebooks = 4
    seq_len = 173
    
    # Create codes with some masked positions
    codes = np.random.randint(0, 1024, size=(batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    # Create mask - True means "generate this position"
    mask = np.zeros((batch_size, n_codebooks, seq_len), dtype=bool)
    # Mask middle portion
    mask[:, :, seq_len//3:2*seq_len//3] = True
    
    print(f"\nTest input shapes:")
    print(f"  codes: {codes.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  Masked positions: {mask.sum()}/{mask.size} ({100*mask.sum()/mask.size:.1f}%)")
    
    # Run inference
    try:
        outputs = session.run(None, {
            'codes': codes,
            'mask': mask
        })
        
        generated = outputs[0]
        print(f"\nOutput shape: {generated.shape}")
        print(f"Output range: [{generated.min()}, {generated.max()}]")
        print(f"Unique values: {len(np.unique(generated))}")
        
        # Check for mask tokens (1024) in output
        mask_tokens = np.sum(generated == 1024)
        print(f"Mask tokens (1024) in output: {mask_tokens}")
        
        # Check if masked positions were changed
        changed = np.sum(codes != generated)
        print(f"Tokens changed: {changed}/{codes.size} ({100*changed/codes.size:.1f}%)")
        
        # Check if only masked positions changed
        unmasked_changed = np.sum((codes != generated) & ~mask)
        masked_changed = np.sum((codes != generated) & mask)
        print(f"Unmasked positions changed: {unmasked_changed}")
        print(f"Masked positions changed: {masked_changed}/{mask.sum()}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        
    # Test C2F transformer
    print("\n\n2. Testing C2F Transformer")
    c2f_path = "scripts/onnx_models_fixed/c2f_transformer_v2_weighted.onnx"
    
    if not Path(c2f_path).exists():
        print(f"C2F transformer not found at {c2f_path}")
        return
        
    c2f_session = ort.InferenceSession(c2f_path)
    
    # C2F expects 14 codebooks (4 coarse + 10 fine)
    full_codes = np.random.randint(0, 1024, size=(batch_size, 14, seq_len), dtype=np.int64)
    full_mask = np.zeros((batch_size, 14, seq_len), dtype=bool)
    # Only mask fine codebooks
    full_mask[:, 4:, :] = True
    
    print(f"\nC2F test input shapes:")
    print(f"  codes: {full_codes.shape}")
    print(f"  mask: {full_mask.shape}")
    
    try:
        c2f_outputs = c2f_session.run(None, {
            'codes': full_codes,
            'mask': full_mask
        })
        
        c2f_generated = c2f_outputs[0]
        print(f"\nC2F output shape: {c2f_generated.shape}")
        print(f"C2F output range: [{c2f_generated.min()}, {c2f_generated.max()}]")
        
        # Check for mask tokens in C2F output
        c2f_mask_tokens = np.sum(c2f_generated == 1024)
        print(f"Mask tokens (1024) in C2F output: {c2f_mask_tokens}")
        
        # Check if coarse codes preserved
        coarse_preserved = np.all(full_codes[:, :4] == c2f_generated[:, :4])
        print(f"Coarse codes preserved: {coarse_preserved}")
        
        # Check fine codes
        fine_changed = np.sum(full_codes[:, 4:] != c2f_generated[:, 4:])
        fine_total = full_codes[:, 4:].size
        print(f"Fine codes changed: {fine_changed}/{fine_total} ({100*fine_changed/fine_total:.1f}%)")
        
    except Exception as e:
        print(f"Error during C2F inference: {e}")


if __name__ == "__main__":
    test_transformer_inference()