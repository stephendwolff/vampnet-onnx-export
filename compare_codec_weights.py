#!/usr/bin/env python3
"""Compare codec weights between VampNet and ONNX encoder."""

import torch
import onnx
import numpy as np
from pathlib import Path
import vampnet

def compare_weights():
    """Compare codec weights between original and ONNX."""
    print("Loading VampNet codec...")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        device='cpu',
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
    )
    
    codec = interface.codec
    
    print("\nLoading ONNX model...")
    onnx_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    model = onnx.load(str(onnx_path))
    
    # Get ONNX weights
    onnx_weights = {}
    for init in model.graph.initializer:
        onnx_weights[init.name] = onnx.numpy_helper.to_array(init)
    
    print("\nComparing codebook weights...")
    print("-" * 80)
    
    mismatches = []
    
    # Check each quantizer
    for i in range(14):
        print(f"\nQuantizer {i}:")
        
        # Get VampNet codebook
        vampnet_codebook = codec.quantizer.quantizers[i].codebook.weight.data.cpu().numpy()
        print(f"  VampNet shape: {vampnet_codebook.shape}")
        print(f"  VampNet stats: mean={vampnet_codebook.mean():.6f}, std={vampnet_codebook.std():.6f}")
        
        # Get ONNX codebook
        onnx_key = f"codec._orig_mod.quantizer.quantizers.{i}.codebook.weight"
        if onnx_key in onnx_weights:
            onnx_codebook = onnx_weights[onnx_key]
            print(f"  ONNX shape: {onnx_codebook.shape}")
            print(f"  ONNX stats: mean={onnx_codebook.mean():.6f}, std={onnx_codebook.std():.6f}")
            
            # Compare
            if vampnet_codebook.shape == onnx_codebook.shape:
                diff = np.abs(vampnet_codebook - onnx_codebook)
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-5:
                    print("  ✅ Weights match!")
                else:
                    print("  ❌ Weights differ!")
                    mismatches.append(i)
                    
                    # Show some specific differences
                    if max_diff > 0.1:
                        print("  Sample differences:")
                        indices = np.unravel_index(diff.argmax(), diff.shape)
                        print(f"    At {indices}: VampNet={vampnet_codebook[indices]:.6f}, ONNX={onnx_codebook[indices]:.6f}")
            else:
                print("  ❌ Shape mismatch!")
                mismatches.append(i)
        else:
            print("  ❌ Not found in ONNX model!")
            mismatches.append(i)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    if not mismatches:
        print("✅ All codebook weights match between VampNet and ONNX!")
    else:
        print(f"❌ Found mismatches in {len(mismatches)} quantizers: {mismatches}")
        print("\nThe ONNX encoder has the correct weights structure but values differ.")
        print("This suggests the codec was exported but weights weren't properly transferred.")


if __name__ == "__main__":
    compare_weights()