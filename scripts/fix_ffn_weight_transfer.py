"""
Fix FFN weight transfer by handling GatedGELU activation properly.
"""

import torch
import torch.nn as nn
import numpy as np
import vampnet
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX
from scripts.transfer_weights_improved import (
    transfer_attention_weights, 
    transfer_output_projections
)


class GatedFFN(nn.Module):
    """FFN with GatedGELU activation matching VampNet."""
    
    def __init__(self, d_model):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)  # 1280 -> 5120
        self.w_2 = nn.Linear(d_model * 2, d_model)  # 2560 -> 1280
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Project to 4*d_model
        hidden = self.w_1(x)
        
        # Split into two halves for gating
        hidden, gate = hidden.chunk(2, dim=-1)  # Each is 2560
        
        # Apply gated activation
        hidden = hidden * self.activation(gate)
        
        # Project back
        return self.w_2(hidden)


def create_vampnet_compatible_model():
    """Create ONNX model with VampNet-compatible FFN."""
    
    print("=== Creating VampNet-Compatible Model ===\n")
    
    class VampNetTransformerONNXFixed(VampNetTransformerONNX):
        """Fixed version with proper FFN dimensions."""
        
        def __init__(self, *args, **kwargs):
            # First initialize parent
            super().__init__(*args, **kwargs)
            
            # Replace FFN modules with GatedFFN
            for i in range(len(self.layers)):
                self.layers[i]['ffn'] = GatedFFN(self.d_model)
    
    model = VampNetTransformerONNXFixed(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    return model


def transfer_weights_with_fixed_ffn():
    """Transfer weights with proper FFN handling."""
    
    print("=== Weight Transfer with Fixed FFN ===\n")
    
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
    
    # Create fixed model
    onnx_model = create_vampnet_compatible_model()
    
    total_mapped = 0
    
    # Transfer embeddings (same as before)
    print("=== Transferring Embeddings ===")
    pos_embed_candidates = [name for name in vampnet_state if 'pos' in name and 'embed' in name]
    for name in pos_embed_candidates:
        param = vampnet_state[name]
        if param.dim() == 3 and param.shape[2] == 1280:
            onnx_model.pos_encoding.data[:, :param.shape[1], :] = param
            print(f"✓ Positional encoding: {name}")
            total_mapped += 1
            break
    
    # Transfer transformer layers
    print("\n=== Transferring Transformer Layers ===")
    for layer_idx in range(20):
        print(f"\nLayer {layer_idx}:")
        
        # Norms
        for norm_idx, norm_type in enumerate(['norm1', 'norm2']):
            norm_name = f'transformer.layers.{layer_idx}.norm_{norm_idx*2 + 1}.weight'
            if norm_name in vampnet_state:
                weight = vampnet_state[norm_name]
                if weight.shape[0] == 1280:
                    getattr(onnx_model.layers[layer_idx], norm_type).weight.data = weight
                    print(f"  ✓ {norm_type}")
                    total_mapped += 1
        
        # Attention
        total_mapped += transfer_attention_weights(layer_idx, vampnet_state, onnx_model)
        
        # FFN with fixed dimensions
        ffn_module = onnx_model.layers[layer_idx]['ffn']
        
        # w_1 weight
        w1_name = f'transformer.layers.{layer_idx}.feed_forward.w_1.weight'
        if w1_name in vampnet_state:
            ffn_module.w_1.weight.data = vampnet_state[w1_name]
            print(f"  ✓ FFN w_1 weight")
            total_mapped += 1
        
        # w_2 weight (now dimensions match!)
        w2_name = f'transformer.layers.{layer_idx}.feed_forward.w_2.weight'
        if w2_name in vampnet_state:
            w2_weight = vampnet_state[w2_name]
            if w2_weight.shape == ffn_module.w_2.weight.shape:
                ffn_module.w_2.weight.data = w2_weight
                print(f"  ✓ FFN w_2 weight (dimensions match!)")
                total_mapped += 1
            else:
                print(f"  ✗ FFN w_2 shape mismatch: {w2_weight.shape} vs {ffn_module.w_2.weight.shape}")
    
    # Final norm
    print("\n=== Transferring Final Norm ===")
    for name in vampnet_state:
        if 'norm' in name and vampnet_state[name].shape == (1280,) and 'layers' not in name:
            onnx_model.final_norm.weight.data = vampnet_state[name]
            print(f"✓ Final norm: {name}")
            total_mapped += 1
            break
    
    # Output projections
    print("\n=== Transferring Output Projections ===")
    total_mapped += transfer_output_projections(vampnet_state, onnx_model)
    
    print(f"\n=== Transfer Summary ===")
    print(f"Total weights mapped: {total_mapped}")
    print(f"Expected mappings: ~141 (with both FFN layers)")
    
    # Save weights
    torch.save(onnx_model.state_dict(), "vampnet_onnx_weights_complete.pth")
    print("\n✓ Saved complete weights to vampnet_onnx_weights_complete.pth")
    
    return onnx_model, total_mapped


def test_and_export_fixed(model):
    """Test and export the fixed model."""
    
    print("\n=== Testing Fixed Model ===")
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
            "vampnet_transformer_complete.onnx",
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
        
        print("✓ Exported to vampnet_transformer_complete.onnx")
        
        # Check file size
        size_mb = os.path.getsize("vampnet_transformer_complete.onnx") / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nNote: GatedFFN might need ONNX-compatible implementation")
        return False


def create_onnx_compatible_gated_ffn():
    """Create ONNX-compatible version of GatedFFN."""
    
    print("\n\n=== ONNX-Compatible GatedFFN ===")
    
    class ONNXGatedFFN(nn.Module):
        """ONNX-compatible FFN with gated activation."""
        
        def __init__(self, d_model):
            super().__init__()
            self.w_1 = nn.Linear(d_model, d_model * 4)
            self.w_2 = nn.Linear(d_model * 2, d_model)
            self.d_model = d_model
            
        def forward(self, x):
            # Project
            hidden = self.w_1(x)
            
            # Split using slice operations (ONNX-friendly)
            d_half = self.d_model * 2
            hidden_part = hidden[:, :, :d_half]
            gate_part = hidden[:, :, d_half:]
            
            # GELU activation on gate
            gate_activated = torch.nn.functional.gelu(gate_part)
            
            # Element-wise multiplication
            gated = hidden_part * gate_activated
            
            # Final projection
            return self.w_2(gated)
    
    print("Created ONNX-compatible GatedFFN")
    print("This version uses explicit slicing instead of chunk()")
    
    return ONNXGatedFFN


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Transfer weights with fixed FFN
    model, mapped = transfer_weights_with_fixed_ffn()
    
    # Test and export
    success = test_and_export_fixed(model)
    
    if not success:
        print("\nTrying with ONNX-compatible GatedFFN...")
        
        # Replace with ONNX-compatible version
        ONNXGatedFFN = create_onnx_compatible_gated_ffn()
        for i in range(len(model.layers)):
            old_ffn = model.layers[i]['ffn']
            new_ffn = ONNXGatedFFN(model.d_model)
            # Copy weights
            new_ffn.w_1.weight.data = old_ffn.w_1.weight.data
            new_ffn.w_2.weight.data = old_ffn.w_2.weight.data
            model.layers[i]['ffn'] = new_ffn
        
        # Try export again
        test_and_export_fixed(model)
    
    print("\n=== Summary ===")
    print(f"Successfully mapped {mapped} weights")
    print("FFN dimensions now match VampNet's GatedGELU structure")
    print("Files created:")
    print("  - vampnet_onnx_weights_complete.pth")
    print("  - vampnet_transformer_complete.onnx (if export succeeded)")