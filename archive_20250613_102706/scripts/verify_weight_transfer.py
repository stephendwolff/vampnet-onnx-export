#!/usr/bin/env python3
"""
Verify if transformer weights were properly transferred.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import vampnet


def verify_weight_transfer():
    """Check if ONNX model weights match VampNet weights."""
    
    print("=== Verifying Weight Transfer ===\n")
    
    # Load VampNet coarse model
    print("1. Loading VampNet coarse model...")
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    vampnet_model = interface.coarse
    vampnet_state = vampnet_model.state_dict()
    
    print(f"VampNet model parameters: {len(vampnet_state)}")
    
    # Sample some key weights
    print("\n2. Sampling VampNet weights:")
    
    # Check embedding weight
    if 'embedding.weight' in vampnet_state:
        emb_weight = vampnet_state['embedding.weight']
        print(f"  Embedding shape: {emb_weight.shape}")
        print(f"  Embedding sample: {emb_weight[0, :5].cpu().numpy()}")
    
    # Check first transformer layer
    for key in vampnet_state.keys():
        if 'layers.0' in key and 'weight' in key:
            print(f"  {key}: shape={vampnet_state[key].shape}")
            if vampnet_state[key].numel() > 0:
                print(f"    Sample: {vampnet_state[key].flatten()[:5].cpu().numpy()}")
            break
    
    # Load ONNX model and check if it produces similar outputs
    print("\n3. Testing ONNX model behavior...")
    
    onnx_path = "onnx_models_fixed/coarse_transformer_v2_weighted.onnx"
    if not Path(onnx_path).exists():
        print(f"ONNX model not found at {onnx_path}")
        return
        
    session = ort.InferenceSession(onnx_path)
    
    # Create identical input
    batch_size = 1
    n_codebooks = 4
    seq_len = 50  # Short sequence for testing
    
    # Create input with known pattern
    codes = np.zeros((batch_size, n_codebooks, seq_len), dtype=np.int64)
    # Fill with a simple pattern
    for i in range(n_codebooks):
        codes[:, i, :] = np.arange(seq_len) % 100 + i * 100
    
    # No masking for this test
    mask = np.zeros_like(codes, dtype=bool)
    
    # Run ONNX
    onnx_output = session.run(None, {'codes': codes, 'mask': mask})[0]
    
    print(f"\nONNX output shape: {onnx_output.shape}")
    print(f"Input vs Output difference: {np.sum(codes != onnx_output)}")
    
    # With masking
    mask[:, :, 10:20] = True
    onnx_output_masked = session.run(None, {'codes': codes, 'mask': mask})[0]
    
    masked_changed = np.sum((codes != onnx_output_masked) & mask)
    unmasked_changed = np.sum((codes != onnx_output_masked) & ~mask)
    
    print(f"\nWith masking:")
    print(f"  Masked positions changed: {masked_changed}/{mask.sum()}")
    print(f"  Unmasked positions changed: {unmasked_changed}")
    
    # Check token distribution in output
    unique_tokens = len(np.unique(onnx_output_masked))
    print(f"  Unique tokens in output: {unique_tokens}")
    print(f"  Output range: [{onnx_output_masked.min()}, {onnx_output_masked.max()}]")
    
    # Compare with VampNet generation
    print("\n4. Comparing with VampNet generation...")
    
    # Convert to torch tensors
    codes_torch = torch.from_numpy(codes).to(vampnet_model.device)
    mask_torch = torch.from_numpy(mask).to(vampnet_model.device).long()
    
    # VampNet uses mask differently (1=keep, 0=mask)
    vampnet_mask = 1 - mask_torch
    
    try:
        with torch.no_grad():
            # Apply masking
            z_masked = codes_torch.clone()
            z_masked[mask_torch.bool()] = vampnet_model.mask_token
            
            # Generate
            vampnet_output = vampnet_model.generate(
                codec=None,
                time_steps=1,
                start_tokens=z_masked,
                mask=vampnet_mask,
                temperature=1.0
            )
            
        print(f"VampNet output shape: {vampnet_output.shape}")
        vampnet_unique = len(torch.unique(vampnet_output))
        print(f"VampNet unique tokens: {vampnet_unique}")
        
    except Exception as e:
        print(f"VampNet generation failed: {e}")
    
    print("\n" + "="*50)
    print("Summary:")
    print("- If ONNX produces very different token distributions than VampNet,")
    print("  the weights were likely not transferred correctly.")
    print("- Check if the unique token counts are similar between models.")
    

if __name__ == "__main__":
    verify_weight_transfer()