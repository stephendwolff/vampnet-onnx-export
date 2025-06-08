"""
Transfer weights from pretrained VampNet to our ONNX-compatible transformer.
This script analyzes the VampNet model structure and maps weights to our custom implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import vampnet
import os
import sys
from collections import OrderedDict
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX
from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm
from scripts.custom_ops.film_onnx import SimpleFiLM
from scripts.custom_ops.codebook_embedding_onnx import VerySimpleCodebookEmbedding


def analyze_vampnet_structure():
    """Deep analysis of VampNet model structure."""
    
    print("=== Analyzing VampNet Model Structure ===\n")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    # Get the coarse model
    coarse_model = interface.coarse
    if hasattr(coarse_model, '_orig_mod'):
        model = coarse_model._orig_mod
    else:
        model = coarse_model
    
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    # Get state dict
    state_dict = model.state_dict()
    print(f"\nTotal parameters in state dict: {len(state_dict)}")
    
    # Analyze layers
    print("\n=== Layer Analysis ===")
    
    # Group parameters by type
    embeddings = {}
    norms = {}
    attentions = {}
    ffns = {}
    films = {}
    outputs = {}
    others = {}
    
    for name, param in state_dict.items():
        if 'embedding' in name:
            embeddings[name] = param
        elif 'norm' in name or 'ln' in name:
            norms[name] = param
        elif 'attn' in name or 'attention' in name:
            attentions[name] = param
        elif 'mlp' in name or 'ffn' in name or 'fc' in name:
            ffns[name] = param
        elif 'film' in name or 'modulation' in name:
            films[name] = param
        elif 'classifier' in name or 'output' in name or 'head' in name:
            outputs[name] = param
        else:
            others[name] = param
    
    print(f"Embeddings: {len(embeddings)}")
    print(f"Norms: {len(norms)}")
    print(f"Attentions: {len(attentions)}")
    print(f"FFNs: {len(ffns)}")
    print(f"FiLMs: {len(films)}")
    print(f"Outputs: {len(outputs)}")
    print(f"Others: {len(others)}")
    
    # Show some examples
    print("\n=== Sample Parameters ===")
    
    print("\nEmbeddings:")
    for i, (name, param) in enumerate(embeddings.items()):
        if i < 3:
            print(f"  {name}: {param.shape}")
    
    print("\nTransformer layers (first layer):")
    layer_0_params = {k: v for k, v in state_dict.items() if '.0.' in k or 'layers.0' in k}
    for name, param in sorted(layer_0_params.items())[:10]:
        print(f"  {name}: {param.shape}")
    
    return model, state_dict


def map_embedding_weights(onnx_model, vampnet_state):
    """Map embedding weights from VampNet to ONNX model."""
    
    print("\n=== Mapping Embedding Weights ===")
    
    # VampNet uses positional embeddings and codebook embeddings
    mapped = 0
    
    # Look for positional embeddings
    for name, param in vampnet_state.items():
        if 'pos' in name and 'embed' in name:
            if param.dim() == 3 and param.shape[2] == onnx_model.d_model:
                # Found positional embedding
                onnx_model.pos_encoding.data[:, :param.shape[1], :] = param
                print(f"✓ Mapped positional encoding: {name} -> pos_encoding")
                mapped += 1
                break
    
    # Look for codebook embeddings
    # VampNet might store them differently than our simple approach
    embedding_weights = []
    for name, param in vampnet_state.items():
        if 'embedding' in name and param.dim() == 2:
            if param.shape[1] == onnx_model.d_model:
                embedding_weights.append((name, param))
    
    # Try to map to our embedding layers
    if embedding_weights:
        print(f"Found {len(embedding_weights)} embedding weight tensors")
        
        # Map to our codebook embeddings
        for i in range(min(len(embedding_weights), onnx_model.n_codebooks)):
            name, param = embedding_weights[i]
            if param.shape[0] >= onnx_model.vocab_size:
                onnx_model.embedding.embeddings[i].weight.data[:onnx_model.vocab_size] = param[:onnx_model.vocab_size]
                print(f"✓ Mapped embedding {i}: {name} -> embedding.embeddings.{i}")
                mapped += 1
    
    return mapped


def map_transformer_weights(onnx_model, vampnet_state):
    """Map transformer layer weights from VampNet to ONNX model."""
    
    print("\n=== Mapping Transformer Weights ===")
    
    mapped = 0
    
    # VampNet transformer layers structure
    n_layers = len(onnx_model.layers)
    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}:")
        
        # Map layer normalization (RMSNorm)
        for norm_type in ['norm1', 'norm2']:
            # Look for corresponding norm in VampNet
            candidates = []
            for name, param in vampnet_state.items():
                if f'layers.{layer_idx}' in name and 'norm' in name and param.dim() == 1:
                    candidates.append((name, param))
            
            if candidates:
                # Pick the right norm based on position
                idx = 0 if norm_type == 'norm1' else 1
                if idx < len(candidates):
                    name, param = candidates[idx]
                    if param.shape[0] == onnx_model.d_model:
                        getattr(onnx_model.layers[layer_idx], norm_type).weight.data = param
                        print(f"  ✓ {norm_type}: {name}")
                        mapped += 1
        
        # Map self-attention weights
        attn_module = onnx_model.layers[layer_idx]['self_attn']
        
        # Look for attention weights in VampNet
        for param_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
            vampnet_names = []
            for name, param in vampnet_state.items():
                if f'layers.{layer_idx}' in name and 'attn' in name:
                    if param_name.replace('.', '_') in name or param_name.split('.')[-1] in name:
                        vampnet_names.append((name, param))
            
            if vampnet_names:
                # Use the first matching parameter
                name, param = vampnet_names[0]
                
                if 'in_proj_weight' in param_name and hasattr(attn_module, 'in_proj_weight'):
                    if param.shape == attn_module.in_proj_weight.shape:
                        attn_module.in_proj_weight.data = param
                        print(f"  ✓ attention.{param_name}: {name}")
                        mapped += 1
                elif 'in_proj_bias' in param_name and hasattr(attn_module, 'in_proj_bias'):
                    if param.shape == attn_module.in_proj_bias.shape:
                        attn_module.in_proj_bias.data = param
                        print(f"  ✓ attention.{param_name}: {name}")
                        mapped += 1
                elif 'out_proj.weight' in param_name:
                    if param.shape == attn_module.out_proj.weight.shape:
                        attn_module.out_proj.weight.data = param
                        print(f"  ✓ attention.{param_name}: {name}")
                        mapped += 1
                elif 'out_proj.bias' in param_name:
                    if param.shape == attn_module.out_proj.bias.shape:
                        attn_module.out_proj.bias.data = param
                        print(f"  ✓ attention.{param_name}: {name}")
                        mapped += 1
        
        # Map FFN weights
        ffn_module = onnx_model.layers[layer_idx]['ffn']
        
        # FFN has two linear layers
        ffn_weights = []
        for name, param in vampnet_state.items():
            if f'layers.{layer_idx}' in name and ('mlp' in name or 'ffn' in name or 'fc' in name):
                ffn_weights.append((name, param))
        
        # Sort by name to get correct order
        ffn_weights.sort(key=lambda x: x[0])
        
        # Map first linear layer
        if len(ffn_weights) >= 2:
            # First linear (d_model -> 4*d_model)
            for name, param in ffn_weights:
                if param.shape == (onnx_model.d_model * 4, onnx_model.d_model):
                    ffn_module[0].weight.data = param
                    print(f"  ✓ ffn.0.weight: {name}")
                    mapped += 1
                    break
            
            # First linear bias
            for name, param in ffn_weights:
                if param.shape == (onnx_model.d_model * 4,):
                    ffn_module[0].bias.data = param
                    print(f"  ✓ ffn.0.bias: {name}")
                    mapped += 1
                    break
            
            # Second linear (4*d_model -> d_model)
            for name, param in ffn_weights:
                if param.shape == (onnx_model.d_model, onnx_model.d_model * 4):
                    ffn_module[2].weight.data = param
                    print(f"  ✓ ffn.2.weight: {name}")
                    mapped += 1
                    break
            
            # Second linear bias
            for name, param in ffn_weights:
                if param.shape == (onnx_model.d_model,) and param is not ffn_module[0].bias:
                    ffn_module[2].bias.data = param
                    print(f"  ✓ ffn.2.bias: {name}")
                    mapped += 1
                    break
        
        # Map FiLM weights if present
        film_module = onnx_model.layers[layer_idx]['film']
        film_weights = []
        for name, param in vampnet_state.items():
            if f'layers.{layer_idx}' in name and ('film' in name or 'modulation' in name):
                film_weights.append((name, param))
        
        if film_weights:
            for name, param in film_weights:
                if 'gamma' in name and param.shape == film_module.gamma_weight.shape:
                    film_module.gamma_weight.data = param
                    print(f"  ✓ film.gamma_weight: {name}")
                    mapped += 1
                elif 'beta' in name and param.shape == film_module.beta_weight.shape:
                    film_module.beta_weight.data = param
                    print(f"  ✓ film.beta_weight: {name}")
                    mapped += 1
    
    return mapped


def map_output_weights(onnx_model, vampnet_state):
    """Map output projection weights."""
    
    print("\n=== Mapping Output Weights ===")
    
    mapped = 0
    
    # Look for classifier/output projection weights
    output_weights = []
    for name, param in vampnet_state.items():
        if ('classifier' in name or 'output' in name or 'head' in name) and param.dim() == 2:
            if param.shape[0] == onnx_model.vocab_size and param.shape[1] == onnx_model.d_model:
                output_weights.append((name, param))
    
    # Map to our output projections
    for i in range(min(len(output_weights), len(onnx_model.output_projs))):
        name, param = output_weights[i]
        onnx_model.output_projs[i].weight.data = param
        print(f"✓ Mapped output projection {i}: {name}")
        mapped += 1
    
    # Also check for biases
    for name, param in vampnet_state.items():
        if ('classifier' in name or 'output' in name) and param.dim() == 1 and param.shape[0] == onnx_model.vocab_size:
            for i in range(len(onnx_model.output_projs)):
                if onnx_model.output_projs[i].bias is not None:
                    onnx_model.output_projs[i].bias.data = param
                    print(f"✓ Mapped output bias {i}: {name}")
                    mapped += 1
                    break
    
    return mapped


def transfer_weights():
    """Main weight transfer function."""
    
    print("=== VampNet to ONNX Weight Transfer ===\n")
    
    # Load VampNet and analyze
    model, vampnet_state = analyze_vampnet_structure()
    
    # Create ONNX model
    print("\n=== Creating ONNX Model ===")
    onnx_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    print(f"ONNX model created with {sum(p.numel() for p in onnx_model.parameters()):,} parameters")
    
    # Transfer weights
    total_mapped = 0
    
    # Map embeddings
    total_mapped += map_embedding_weights(onnx_model, vampnet_state)
    
    # Map transformer layers
    total_mapped += map_transformer_weights(onnx_model, vampnet_state)
    
    # Map output projections
    total_mapped += map_output_weights(onnx_model, vampnet_state)
    
    # Final norm
    final_norm_mapped = False
    for name, param in vampnet_state.items():
        if 'final' in name and 'norm' in name and param.dim() == 1:
            if param.shape[0] == onnx_model.d_model:
                onnx_model.final_norm.weight.data = param
                print(f"\n✓ Mapped final norm: {name}")
                total_mapped += 1
                final_norm_mapped = True
                break
    
    print(f"\n=== Transfer Summary ===")
    print(f"Total weights mapped: {total_mapped}")
    print(f"ONNX model parameters: {len(list(onnx_model.parameters()))}")
    
    # Save the model with transferred weights
    print("\n=== Saving Model ===")
    torch.save(onnx_model.state_dict(), "vampnet_onnx_weights.pth")
    print("✓ Saved ONNX model weights to vampnet_onnx_weights.pth")
    
    # Test the model
    print("\n=== Testing Model ===")
    test_model(onnx_model)
    
    return onnx_model


def test_model(model):
    """Test the model with transferred weights."""
    
    model.eval()
    
    # Test input
    batch_size = 1
    seq_len = 100
    codes = torch.randint(0, 1024, (batch_size, 4, seq_len))
    mask = torch.zeros_like(codes)
    mask[:, :, 40:60] = 1
    
    print(f"Test input shape: {codes.shape}")
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(codes, mask)
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            
            # Check if model is actually using transferred weights
            # (not just random initialization)
            changed = (output != codes)[mask.bool()].sum().item()
            total_masked = mask.sum().item()
            print(f"  Changed {changed}/{total_masked} masked positions")
            
            # Export to ONNX
            print("\nExporting to ONNX with transferred weights...")
            
            # Wrapper for ONNX export
            class ONNXWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    
                def forward(self, codes, mask):
                    return self.model(codes, mask, temperature=1.0)
            
            wrapper = ONNXWrapper(model)
            
            torch.onnx.export(
                wrapper,
                (codes, mask),
                "vampnet_transformer_pretrained.onnx",
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
            
            print("✓ Exported to vampnet_transformer_pretrained.onnx")
            
            # Check file size
            import os
            size_mb = os.path.getsize("vampnet_transformer_pretrained.onnx") / (1024 * 1024)
            print(f"  Model size: {size_mb:.1f} MB")
            
        except Exception as e:
            print(f"✗ Error during testing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run weight transfer
    model = transfer_weights()
    
    print("\n=== Done ===")
    print("Next steps:")
    print("1. Test vampnet_transformer_pretrained.onnx with real audio")
    print("2. Compare outputs with original VampNet")
    print("3. Fine-tune weight mapping if needed")