"""
Improved weight transfer from VampNet to ONNX with proper attention weight handling.
"""

import torch
import torch.nn as nn
import numpy as np
import vampnet
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX


def transfer_attention_weights(layer_idx, vampnet_state, onnx_model):
    """Transfer attention weights with proper concatenation."""
    
    mapped = 0
    attn_module = onnx_model.layers[layer_idx]['self_attn']
    
    # Get Q, K, V weights from VampNet
    q_weight_name = f'transformer.layers.{layer_idx}.self_attn.w_qs.weight'
    k_weight_name = f'transformer.layers.{layer_idx}.self_attn.w_ks.weight'
    v_weight_name = f'transformer.layers.{layer_idx}.self_attn.w_vs.weight'
    
    if all(name in vampnet_state for name in [q_weight_name, k_weight_name, v_weight_name]):
        q_weight = vampnet_state[q_weight_name]
        k_weight = vampnet_state[k_weight_name]
        v_weight = vampnet_state[v_weight_name]
        
        # Concatenate Q, K, V weights for in_proj_weight
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        if hasattr(attn_module, 'in_proj_weight') and in_proj_weight.shape == attn_module.in_proj_weight.shape:
            attn_module.in_proj_weight.data = in_proj_weight
            print(f"  ✓ Attention in_proj_weight: concatenated Q,K,V weights")
            mapped += 1
    
    # Transfer output projection
    out_proj_name = f'transformer.layers.{layer_idx}.self_attn.fc.weight'
    if out_proj_name in vampnet_state:
        out_weight = vampnet_state[out_proj_name]
        if out_weight.shape == attn_module.out_proj.weight.shape:
            attn_module.out_proj.weight.data = out_weight
            print(f"  ✓ Attention out_proj.weight")
            mapped += 1
    
    return mapped


def transfer_ffn_weights(layer_idx, vampnet_state, onnx_model):
    """Transfer FFN weights with correct naming."""
    
    mapped = 0
    ffn_module = onnx_model.layers[layer_idx]['ffn']
    
    # First linear layer (d_model -> 4*d_model)
    # VampNet uses w_1, but also check for different naming
    w1_names = [
        f'transformer.layers.{layer_idx}.feed_forward.w_1.weight',
        f'transformer.layers.{layer_idx}.mlp.w_1.weight',
        f'transformer.layers.{layer_idx}.ffn.fc1.weight'
    ]
    
    for name in w1_names:
        if name in vampnet_state:
            weight = vampnet_state[name]
            # Check if dimensions match (might need transpose)
            if weight.shape == ffn_module[0].weight.shape:
                ffn_module[0].weight.data = weight
                print(f"  ✓ FFN first layer weight: {name}")
                mapped += 1
                break
            elif weight.T.shape == ffn_module[0].weight.shape:
                ffn_module[0].weight.data = weight.T
                print(f"  ✓ FFN first layer weight (transposed): {name}")
                mapped += 1
                break
    
    # Second linear layer (4*d_model -> d_model)
    w2_names = [
        f'transformer.layers.{layer_idx}.feed_forward.w_2.weight',
        f'transformer.layers.{layer_idx}.mlp.w_2.weight',
        f'transformer.layers.{layer_idx}.ffn.fc2.weight'
    ]
    
    for name in w2_names:
        if name in vampnet_state:
            weight = vampnet_state[name]
            if weight.shape == ffn_module[2].weight.shape:
                ffn_module[2].weight.data = weight
                print(f"  ✓ FFN second layer weight: {name}")
                mapped += 1
                break
            elif weight.T.shape == ffn_module[2].weight.shape:
                ffn_module[2].weight.data = weight.T
                print(f"  ✓ FFN second layer weight (transposed): {name}")
                mapped += 1
                break
    
    return mapped


def transfer_output_projections(vampnet_state, onnx_model):
    """Transfer output projections handling weight normalization."""
    
    mapped = 0
    
    # VampNet uses weight normalization for classifiers
    for i in range(len(onnx_model.output_projs)):
        weight_v_name = f'classifier.layers.{i}.weight_v'
        weight_g_name = f'classifier.layers.{i}.weight_g'
        bias_name = f'classifier.layers.{i}.bias'
        
        if weight_v_name in vampnet_state and weight_g_name in vampnet_state:
            weight_v = vampnet_state[weight_v_name]
            weight_g = vampnet_state[weight_g_name]
            
            # Compute normalized weight
            # weight = weight_v * (weight_g / ||weight_v||)
            # Note: weight_v has shape [out, in, 1] in VampNet
            weight_v = weight_v.squeeze(-1)  # Remove last dimension
            norm = torch.norm(weight_v, dim=1, keepdim=True)
            weight = weight_v * (weight_g.squeeze(-1) / norm)
            
            if weight.shape == onnx_model.output_projs[i].weight.shape:
                onnx_model.output_projs[i].weight.data = weight
                print(f"✓ Output projection {i} weight (from weight norm)")
                mapped += 1
        
        # Transfer bias
        if bias_name in vampnet_state:
            bias = vampnet_state[bias_name]
            if bias.shape == onnx_model.output_projs[i].bias.shape:
                onnx_model.output_projs[i].bias.data = bias
                print(f"✓ Output projection {i} bias")
                mapped += 1
    
    return mapped


def improved_weight_transfer():
    """Improved weight transfer with better mapping."""
    
    print("=== Improved VampNet to ONNX Weight Transfer ===\n")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    vampnet_model = interface.coarse
    if hasattr(vampnet_model, '_orig_mod'):
        vampnet_model = vampnet_model._orig_mod
    
    vampnet_state = vampnet_model.state_dict()
    
    # Create ONNX model
    onnx_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    total_mapped = 0
    
    # 1. Transfer embeddings
    print("=== Transferring Embeddings ===")
    # Positional encoding
    pos_embed_candidates = [name for name in vampnet_state if 'pos' in name and 'embed' in name]
    for name in pos_embed_candidates:
        param = vampnet_state[name]
        if param.dim() == 3 and param.shape[2] == 1280:
            onnx_model.pos_encoding.data[:, :param.shape[1], :] = param
            print(f"✓ Positional encoding: {name}")
            total_mapped += 1
            break
    
    # 2. Transfer transformer layers
    print("\n=== Transferring Transformer Layers ===")
    for layer_idx in range(20):
        print(f"\nLayer {layer_idx}:")
        
        # RMSNorm weights (already working)
        for norm_idx, norm_type in enumerate(['norm1', 'norm2']):
            norm_names = [
                f'transformer.layers.{layer_idx}.norm_{norm_idx*2 + 1}.weight',
                f'transformer.layers.{layer_idx}.{norm_type}.weight'
            ]
            
            for name in norm_names:
                if name in vampnet_state:
                    weight = vampnet_state[name]
                    if weight.shape[0] == 1280:
                        getattr(onnx_model.layers[layer_idx], norm_type).weight.data = weight
                        print(f"  ✓ {norm_type}: {name}")
                        total_mapped += 1
                        break
        
        # Attention weights (improved)
        total_mapped += transfer_attention_weights(layer_idx, vampnet_state, onnx_model)
        
        # FFN weights (improved)
        total_mapped += transfer_ffn_weights(layer_idx, vampnet_state, onnx_model)
    
    # 3. Final norm
    print("\n=== Transferring Final Norm ===")
    final_norm_names = ['transformer.norm.weight', 'final_norm.weight', 'ln_f.weight']
    for name in vampnet_state:
        if any(fn in name for fn in ['final', 'norm']) and vampnet_state[name].shape == (1280,):
            onnx_model.final_norm.weight.data = vampnet_state[name]
            print(f"✓ Final norm: {name}")
            total_mapped += 1
            break
    
    # 4. Output projections (improved)
    print("\n=== Transferring Output Projections ===")
    total_mapped += transfer_output_projections(vampnet_state, onnx_model)
    
    print(f"\n=== Transfer Summary ===")
    print(f"Total weights mapped: {total_mapped}")
    print(f"ONNX model parameters: {len(list(onnx_model.parameters()))}")
    
    # Save improved weights
    torch.save(onnx_model.state_dict(), "vampnet_onnx_weights_improved.pth")
    print("\n✓ Saved improved weights to vampnet_onnx_weights_improved.pth")
    
    # Test and export
    test_and_export(onnx_model)
    
    return onnx_model


def test_and_export(model):
    """Test model and export to ONNX."""
    
    print("\n=== Testing Model ===")
    model.eval()
    
    # Test input
    codes = torch.randint(0, 1024, (1, 4, 100))
    mask = torch.zeros_like(codes)
    mask[:, :, 40:60] = 1
    
    with torch.no_grad():
        output = model(codes, mask)
        print(f"✓ Forward pass successful!")
        changed = (output != codes)[mask.bool()].sum().item()
        print(f"  Changed {changed}/{mask.sum().item()} masked positions")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, codes, mask):
            return self.model(codes, mask, temperature=1.0)
    
    wrapper = ONNXWrapper(model)
    
    try:
        torch.onnx.export(
            wrapper,
            (codes, mask),
            "vampnet_transformer_improved.onnx",
            input_names=['codes', 'mask'],
            output_names=['generated_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'mask': {0: 'batch', 2: 'sequence'},
                'generated_codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=13,
            verbose=False
        )
        
        print("✓ Exported to vampnet_transformer_improved.onnx")
        
        # Check file size
        size_mb = os.path.getsize("vampnet_transformer_improved.onnx") / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run improved transfer
    model = improved_weight_transfer()
    
    print("\n=== Done ===")
    print("Files created:")
    print("  - vampnet_onnx_weights_improved.pth")
    print("  - vampnet_transformer_improved.onnx")
    print("\nNext: Test with real audio generation!")