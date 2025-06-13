#!/usr/bin/env python3
"""
Check if the ONNX model weights match VampNet weights.
"""

import torch
import numpy as np
import onnx
from pathlib import Path
import vampnet


def check_weight_similarity():
    """Compare ONNX and VampNet model weights."""
    
    print("=== Checking Model Weight Similarity ===\n")
    
    # Load VampNet
    print("1. Loading VampNet models...")
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth"
    )
    
    # Get VampNet coarse model weights
    vampnet_coarse = interface.coarse
    vampnet_state = vampnet_coarse.state_dict()
    
    print(f"VampNet coarse model has {len(vampnet_state)} parameters")
    
    # Load ONNX model
    print("\n2. Loading ONNX model...")
    onnx_path = Path("scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx")
    onnx_model = onnx.load(str(onnx_path))
    
    # Extract ONNX weights
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = onnx.numpy_helper.to_array(init)
    
    print(f"ONNX model has {len(onnx_weights)} weight tensors")
    
    # Sample some weights to compare
    print("\n3. Sampling weights for comparison...")
    
    # Look for transformer weights
    print("\nTransformer layer weights:")
    for name in sorted(onnx_weights.keys()):
        if 'transformer' in name and 'weight' in name:
            print(f"  {name}: {onnx_weights[name].shape}")
            if len(onnx_weights.keys()) < 50:  # Only show if not too many
                break
    
    # Check embedding dimensions
    print("\n4. Checking embeddings...")
    
    # VampNet embedding info
    vampnet_embed_dim = None
    for name, param in vampnet_state.items():
        if 'embed' in name.lower() and param.dim() >= 2:
            print(f"VampNet {name}: {param.shape}")
            if vampnet_embed_dim is None and param.shape[-1] in [1024, 1280, 512]:
                vampnet_embed_dim = param.shape[-1]
    
    # ONNX embedding info
    onnx_embed_shapes = []
    for name, weight in onnx_weights.items():
        if 'embed' in name.lower() or 'emb' in name.lower():
            print(f"ONNX {name}: {weight.shape}")
            onnx_embed_shapes.append(weight.shape)
    
    # Check for dimension mismatches
    print("\n5. Checking for common issues...")
    
    # Check if embeddings exist
    has_embeddings = any('embed' in name.lower() for name in onnx_weights.keys())
    print(f"  ONNX has embeddings: {has_embeddings}")
    
    # Check classifier/output projection
    has_classifier = any('classifier' in name.lower() or 'output' in name.lower() 
                        for name in onnx_weights.keys())
    print(f"  ONNX has output classifier: {has_classifier}")
    
    # Look for specific layer patterns
    print("\n6. Layer name patterns:")
    
    # VampNet patterns
    vampnet_patterns = set()
    for name in vampnet_state.keys():
        if 'layers' in name:
            # Extract layer pattern
            parts = name.split('.')
            if len(parts) > 2:
                pattern = '.'.join(parts[:3])
                vampnet_patterns.add(pattern)
    
    print(f"\nVampNet layer patterns ({len(vampnet_patterns)} unique):")
    for p in sorted(list(vampnet_patterns))[:5]:
        print(f"  {p}")
    
    # ONNX patterns
    onnx_patterns = set()
    for name in onnx_weights.keys():
        if 'layers' in name:
            parts = name.split('.')
            if len(parts) > 2:
                pattern = '.'.join(parts[:3])
                onnx_patterns.add(pattern)
    
    print(f"\nONNX layer patterns ({len(onnx_patterns)} unique):")
    for p in sorted(list(onnx_patterns))[:5]:
        print(f"  {p}")
    
    # Check weight initialization
    print("\n7. Weight statistics comparison:")
    
    # Sample a few weights
    sample_weights = 5
    weight_count = 0
    
    for onnx_name, onnx_weight in onnx_weights.items():
        if 'weight' in onnx_name and onnx_weight.size > 100:
            weight_count += 1
            if weight_count > sample_weights:
                break
                
            # Find corresponding VampNet weight
            vampnet_name = None
            for vn in vampnet_state.keys():
                if any(part in vn for part in onnx_name.split('.')):
                    vampnet_name = vn
                    break
            
            print(f"\n  {onnx_name}:")
            print(f"    ONNX: mean={onnx_weight.mean():.4f}, std={onnx_weight.std():.4f}")
            
            if vampnet_name and vampnet_name in vampnet_state:
                vw = vampnet_state[vampnet_name].cpu().numpy()
                print(f"    VampNet ({vampnet_name}): mean={vw.mean():.4f}, std={vw.std():.4f}")
    
    return vampnet_state, onnx_weights


if __name__ == "__main__":
    vampnet_state, onnx_weights = check_weight_similarity()