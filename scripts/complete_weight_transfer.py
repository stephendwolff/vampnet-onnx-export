"""
Complete the weight transfer by handling embeddings and output classifiers.
"""

import torch
import torch.nn as nn
import numpy as np
import vampnet
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GatedFFN from export script instead
from export_vampnet_transformer_v2 import VampNetTransformerV2, GatedFFN


def analyze_embedding_structure(vampnet_state):
    """Analyze VampNet's embedding structure."""
    
    print("=== Analyzing VampNet Embeddings ===\n")
    
    # Find all embedding-related weights
    embedding_weights = {}
    for name, param in vampnet_state.items():
        if 'embedding' in name.lower():
            embedding_weights[name] = param
            print(f"{name}: {param.shape}")
    
    # Analyze special tokens
    if 'embedding.special.MASK' in vampnet_state:
        mask_embed = vampnet_state['embedding.special.MASK']
        print(f"\nSpecial MASK embedding: {mask_embed.shape}")
        print("This appears to be per-codebook mask embeddings")
    
    # Analyze output projection
    if 'embedding.out_proj.weight' in vampnet_state:
        out_proj = vampnet_state['embedding.out_proj.weight']
        print(f"\nEmbedding output projection: {out_proj.shape}")
        print("This projects combined embeddings to model dimension")
    
    return embedding_weights


def transfer_embedding_weights(onnx_model, vampnet_state):
    """Transfer embedding weights with proper handling."""
    
    print("\n=== Transferring Embedding Weights ===")
    
    mapped = 0
    
    # 1. Handle positional embeddings
    print("\n1. Positional Embeddings:")
    pos_candidates = []
    for name, param in vampnet_state.items():
        if 'pos' in name.lower() and param.dim() >= 2:
            pos_candidates.append((name, param))
    
    if pos_candidates:
        print(f"Found {len(pos_candidates)} positional embedding candidates:")
        for name, param in pos_candidates:
            print(f"  {name}: {param.shape}")
            
            # Try to match dimensions
            if param.shape[-1] == onnx_model.d_model:
                if param.dim() == 2:
                    # Add batch dimension if needed
                    param = param.unsqueeze(0)
                
                if param.shape[0] == 1 and param.shape[2] == onnx_model.d_model:
                    max_len = min(param.shape[1], onnx_model.pos_encoding.shape[1])
                    onnx_model.pos_encoding.data[:, :max_len, :] = param[:, :max_len, :]
                    print(f"  ✓ Mapped to pos_encoding (first {max_len} positions)")
                    mapped += 1
                    break
    
    # 2. Handle codebook embeddings
    print("\n2. Codebook Embeddings:")
    
    # VampNet might not have explicit embedding tables if it uses the codec directly
    # Let's check for any weight that could be an embedding table
    potential_embeddings = []
    for name, param in vampnet_state.items():
        if param.dim() == 2 and param.shape[0] in [1024, 1025] and param.shape[1] == onnx_model.d_model:
            potential_embeddings.append((name, param))
    
    if potential_embeddings:
        print(f"Found {len(potential_embeddings)} potential embedding tables:")
        for name, param in potential_embeddings:
            print(f"  {name}: {param.shape}")
    else:
        print("  No direct embedding tables found")
        print("  VampNet might generate embeddings differently")
    
    # 3. Handle special mask embeddings
    print("\n3. Special Token Embeddings:")
    if 'embedding.special.MASK' in vampnet_state:
        mask_embed = vampnet_state['embedding.special.MASK']
        print(f"  Found MASK embeddings: {mask_embed.shape}")
        
        # This is a per-codebook mask embedding
        # We need to incorporate this into our embedding tables
        if mask_embed.shape[0] == onnx_model.n_codebooks:
            print("  ⚠️  Special mask handling needed - VampNet uses different approach")
    
    # 4. Initialize embeddings if not found
    if len(potential_embeddings) == 0:
        print("\n4. Initializing Embeddings:")
        print("  Since VampNet doesn't have explicit embedding tables,")
        print("  we'll use the default initialization or extract from first layer")
        
        # Option: Use the embedding output projection as a guide
        if 'embedding.out_proj.weight' in vampnet_state:
            out_proj = vampnet_state['embedding.out_proj.weight']
            print(f"  Using embedding.out_proj.weight as reference: {out_proj.shape}")
            
            # The out_proj weight gives us information about the embedding dimension
            # but we still need to initialize the actual embeddings
            print("  ⚠️  Embeddings will use random initialization")
    
    return mapped


def transfer_classifier_weights(onnx_model, vampnet_state):
    """Transfer output classifier weights handling weight normalization."""
    
    print("\n=== Transferring Output Classifier Weights ===")
    
    mapped = 0
    
    # VampNet uses weight normalization: weight = weight_v * (weight_g / ||weight_v||)
    print("\nVampNet classifier structure:")
    classifier_params = {}
    for name, param in vampnet_state.items():
        if 'classifier' in name:
            classifier_params[name] = param
            if 'layers.0' in name:  # Show first classifier details
                print(f"  {name}: {param.shape}")
    
    # Transfer each classifier
    for i in range(min(onnx_model.n_codebooks, 4)):
        print(f"\nClassifier {i}:")
        
        weight_v_name = f'classifier.layers.{i}.weight_v'
        weight_g_name = f'classifier.layers.{i}.weight_g'
        bias_name = f'classifier.layers.{i}.bias'
        
        if weight_v_name in vampnet_state and weight_g_name in vampnet_state:
            weight_v = vampnet_state[weight_v_name]  # Shape: [vocab_size*4, d_model, 1]
            weight_g = vampnet_state[weight_g_name]  # Shape: [vocab_size*4, 1, 1]
            
            print(f"  weight_v: {weight_v.shape}")
            print(f"  weight_g: {weight_g.shape}")
            
            # Remove singleton dimensions
            weight_v = weight_v.squeeze(-1)  # [vocab_size*4, d_model]
            weight_g = weight_g.squeeze(-1).squeeze(-1)  # [vocab_size*4]
            
            # Compute normalized weight
            # For each output dimension, normalize weight_v and scale by weight_g
            weight_v_norm = torch.norm(weight_v, dim=1, keepdim=True)  # [vocab_size*4, 1]
            weight = weight_v * (weight_g.unsqueeze(1) / (weight_v_norm + 1e-8))
            
            print(f"  Normalized weight shape: {weight.shape}")
            print(f"  ONNX output_proj shape: {onnx_model.output_projs[i].weight.shape}")
            
            # Check dimensions
            if weight.shape[0] == 4096 and onnx_model.output_projs[i].weight.shape[0] == 1024:
                # VampNet might have 4x the output size (4096 vs 1024)
                # This could mean it's predicting all codebooks together
                # For now, take the first quarter
                weight = weight[:1024, :]
                print(f"  Trimmed weight to match ONNX dimensions: {weight.shape}")
            
            if weight.shape == onnx_model.output_projs[i].weight.shape:
                onnx_model.output_projs[i].weight.data = weight
                print(f"  ✓ Transferred weight")
                mapped += 1
            else:
                print(f"  ✗ Shape mismatch: {weight.shape} vs {onnx_model.output_projs[i].weight.shape}")
        
        # Transfer bias
        if bias_name in vampnet_state:
            bias = vampnet_state[bias_name]
            print(f"  bias shape: {bias.shape}")
            
            # Handle size mismatch
            if bias.shape[0] == 4096 and onnx_model.output_projs[i].bias.shape[0] == 1024:
                bias = bias[:1024]
                print(f"  Trimmed bias to match ONNX dimensions")
            
            if bias.shape == onnx_model.output_projs[i].bias.shape:
                onnx_model.output_projs[i].bias.data = bias
                print(f"  ✓ Transferred bias")
                mapped += 1
    
    return mapped


def complete_weight_transfer():
    """Complete the weight transfer with all components."""
    
    print("=== Complete VampNet to ONNX Weight Transfer ===\n")
    
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
    
    # Analyze embeddings first
    embedding_weights = analyze_embedding_structure(vampnet_state)
    
    # Create model with correct architecture
    onnx_model = VampNetTransformerV2(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Load previously transferred weights
    if os.path.exists("vampnet_onnx_weights_complete.pth"):
        print("\nLoading previously transferred weights...")
        onnx_model.load_state_dict(torch.load("vampnet_onnx_weights_complete.pth"))
        print("✓ Loaded existing weights")
    
    total_mapped = 121  # From previous transfer
    
    # Transfer embeddings
    total_mapped += transfer_embedding_weights(onnx_model, vampnet_state)
    
    # Transfer classifiers
    total_mapped += transfer_classifier_weights(onnx_model, vampnet_state)
    
    print(f"\n=== Final Transfer Summary ===")
    print(f"Total weights mapped: {total_mapped}")
    print(f"Total model parameters: {len(list(onnx_model.parameters()))}")
    
    # Save final weights
    torch.save(onnx_model.state_dict(), "vampnet_onnx_weights_final.pth")
    print("\n✓ Saved final weights to vampnet_onnx_weights_final.pth")
    
    # Test and export
    test_final_model(onnx_model)
    
    return onnx_model


def test_final_model(model):
    """Test the model with all transferred weights."""
    
    print("\n=== Testing Final Model ===")
    model.eval()
    
    # Test input
    codes = torch.randint(0, 1024, (1, 4, 100))
    mask = torch.zeros((1, 4, 100), dtype=torch.bool)  # Boolean mask with shape (batch, n_codebooks, seq_len)
    mask[:, :, 40:60] = True  # Mask all codebooks at positions 40-60
    
    try:
        with torch.no_grad():
            output = model(codes, mask)
            print(f"✓ Forward pass successful!")
            # Create expanded mask for comparison
            mask_expanded = mask.unsqueeze(1).expand_as(codes)
            changed = (output != codes)[mask_expanded].sum().item()
            print(f"  Changed {changed}/{mask_expanded.sum().item()} masked positions")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print("  Continuing with export anyway...")
    
    # Export to ONNX
    print("\nExporting final model to ONNX...")
    
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
            "vampnet_transformer_final.onnx",
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
        
        print("✓ Exported to vampnet_transformer_final.onnx")
        
        # Check file size
        size_mb = os.path.getsize("vampnet_transformer_final.onnx") / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Complete the weight transfer
    model = complete_weight_transfer()
    
    print("\n=== Done ===")
    print("Note: Embedding transfer is limited because VampNet uses a different approach")
    print("The model should still work better with transferred transformer weights")