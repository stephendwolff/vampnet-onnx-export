"""
Analyze weight mapping between VampNet and ONNX models to improve transfer.
"""

import torch
import vampnet
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX


def detailed_model_analysis():
    """Perform detailed analysis of both models."""
    
    print("=== Detailed Model Analysis ===\n")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    vampnet_model = interface.coarse
    if hasattr(vampnet_model, '_orig_mod'):
        vampnet_model = vampnet_model._orig_mod
    
    # Create ONNX model
    onnx_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Get state dicts
    vampnet_state = vampnet_model.state_dict()
    onnx_state = onnx_model.state_dict()
    
    print("=== VampNet Model Structure ===")
    print(f"Total parameters: {len(vampnet_state)}")
    
    # Group VampNet parameters by component
    vampnet_groups = defaultdict(list)
    for name, param in vampnet_state.items():
        parts = name.split('.')
        if len(parts) > 0:
            component = parts[0]
            vampnet_groups[component].append((name, param.shape))
    
    for component, params in vampnet_groups.items():
        print(f"\n{component}: {len(params)} parameters")
        if len(params) <= 5:
            for name, shape in params:
                print(f"  {name}: {shape}")
    
    print("\n=== ONNX Model Structure ===")
    print(f"Total parameters: {len(onnx_state)}")
    
    # Group ONNX parameters
    onnx_groups = defaultdict(list)
    for name, param in onnx_state.items():
        parts = name.split('.')
        if len(parts) > 0:
            component = parts[0]
            onnx_groups[component].append((name, param.shape))
    
    for component, params in onnx_groups.items():
        print(f"\n{component}: {len(params)} parameters")
        if len(params) <= 5:
            for name, shape in params:
                print(f"  {name}: {shape}")
    
    # Analyze specific components
    print("\n=== Component Analysis ===")
    
    # Embedding analysis
    print("\nEmbedding components:")
    print("VampNet:")
    for name, shape in vampnet_state.items():
        if 'embedding' in name.lower():
            print(f"  {name}: {shape}")
    
    print("\nONNX:")
    for name, shape in onnx_state.items():
        if 'embedding' in name.lower():
            print(f"  {name}: {shape}")
    
    # Attention analysis
    print("\n\nAttention weights (Layer 0):")
    print("VampNet:")
    for name, shape in vampnet_state.items():
        if 'layers.0' in name and 'attn' in name:
            print(f"  {name}: {shape}")
    
    print("\nONNX:")
    for name, shape in onnx_state.items():
        if 'layers.0' in name and 'attn' in name:
            print(f"  {name}: {shape}")
    
    # FFN analysis
    print("\n\nFFN weights (Layer 0):")
    print("VampNet:")
    for name, shape in vampnet_state.items():
        if 'layers.0' in name and ('feed_forward' in name or 'mlp' in name or 'ffn' in name):
            print(f"  {name}: {shape}")
    
    print("\nONNX:")
    for name, shape in onnx_state.items():
        if 'layers.0' in name and 'ffn' in name:
            print(f"  {name}: {shape}")
    
    return vampnet_state, onnx_state


def find_unmapped_weights(vampnet_state, onnx_state):
    """Find weights that haven't been mapped yet."""
    
    print("\n\n=== Finding Unmapped Weights ===")
    
    # Load current mapping
    if os.path.exists("vampnet_onnx_weights.pth"):
        saved_state = torch.load("vampnet_onnx_weights.pth")
        
        # Find which ONNX weights are still random
        unmapped = []
        for name, param in onnx_state.items():
            if name in saved_state:
                # Check if it's still close to initialization
                saved_param = saved_state[name]
                if torch.allclose(param, saved_param, rtol=1e-5):
                    # It was saved, so probably mapped
                    continue
            unmapped.append(name)
        
        print(f"\nUnmapped ONNX parameters: {len(unmapped)}")
        for name in unmapped[:20]:  # Show first 20
            print(f"  {name}: {onnx_state[name].shape}")
    
    # Find potential matches
    print("\n\n=== Potential Weight Matches ===")
    
    # Look for attention in_proj weights
    print("\nAttention in_proj weights:")
    for layer_idx in range(20):
        onnx_name = f'layers.{layer_idx}.self_attn.in_proj_weight'
        if onnx_name in onnx_state:
            onnx_shape = onnx_state[onnx_name].shape
            print(f"\nONNX {onnx_name}: {onnx_shape}")
            
            # Look for matching shapes in VampNet
            for vn_name, vn_param in vampnet_state.items():
                if f'layers.{layer_idx}' in vn_name and 'attn' in vn_name:
                    if vn_param.shape[0] == 1280 or vn_param.shape[1] == 1280:
                        print(f"  Potential match: {vn_name}: {vn_param.shape}")
    
    # Look for FFN weights
    print("\n\nFFN weights:")
    for layer_idx in range(20):
        # First linear layer in FFN
        onnx_name = f'layers.{layer_idx}.ffn.0.weight'
        if onnx_name in onnx_state:
            onnx_shape = onnx_state[onnx_name].shape
            print(f"\nONNX {onnx_name}: {onnx_shape}")
            
            # Look for matching shapes in VampNet
            for vn_name, vn_param in vampnet_state.items():
                if f'layers.{layer_idx}' in vn_name and vn_param.shape == onnx_shape:
                    print(f"  Exact match: {vn_name}")


def suggest_mapping_improvements():
    """Suggest improvements to weight mapping."""
    
    print("\n\n=== Mapping Improvement Suggestions ===")
    
    print("\n1. Attention weights:")
    print("   - VampNet uses separate w_qs, w_ks, w_vs instead of in_proj_weight")
    print("   - Need to concatenate these weights for ONNX's in_proj_weight")
    print("   - Formula: in_proj_weight = torch.cat([w_q, w_k, w_v], dim=0)")
    
    print("\n2. FFN weights:")
    print("   - VampNet stores as 'feed_forward.w_1.weight' and 'feed_forward.w_2.weight'")
    print("   - ONNX expects 'ffn.0.weight' and 'ffn.2.weight'")
    print("   - Direct mapping should work with correct names")
    
    print("\n3. Embedding weights:")
    print("   - VampNet has special MASK embeddings")
    print("   - Need to properly initialize embedding tables")
    print("   - Consider the out_proj convolution weight")
    
    print("\n4. Output projections:")
    print("   - VampNet uses 'classifier.layers.X.weight_v' (with weight normalization)")
    print("   - Need to combine weight_v and weight_g for final weight")
    print("   - Formula: weight = weight_v * (weight_g / ||weight_v||)")


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Detailed analysis
    vampnet_state, onnx_state = detailed_model_analysis()
    
    # Find unmapped weights
    find_unmapped_weights(vampnet_state, onnx_state)
    
    # Suggest improvements
    suggest_mapping_improvements()
    
    print("\n\n=== Summary ===")
    print("Key findings:")
    print("1. Only 60/294 weights were mapped (mostly norms)")
    print("2. Attention weights need concatenation (w_q, w_k, w_v -> in_proj)")
    print("3. FFN weights exist but with different names")
    print("4. Output projections use weight normalization")
    print("5. Embeddings need special handling for mask tokens")