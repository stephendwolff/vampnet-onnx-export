#!/usr/bin/env python3
"""
Final test of V11 with correct output comparison.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm
from scripts.export_vampnet_transformer_v11_fixed import VampNetTransformerV11, transfer_weights_v11

print("Final V11 test with correct comparison...")

# Load models
torch.manual_seed(42)
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
remove_weight_norm(vampnet.classifier.layers[0])
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

model = VampNetTransformerV11()
transfer_weights_v11("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test
torch.manual_seed(42)
codes = torch.randint(0, 1024, (1, 4, 10))
masked_codes = codes.clone()
masked_codes[:, :, 5:] = 1024

with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Get outputs
    vampnet_out = vampnet(latents)
    v11_out = model(latents)
    
    print(f"\nVampNet output: {vampnet_out.shape}")  # [1, 1024, 40]
    print(f"V11 output: {v11_out.shape}")            # [1, 4, 10, 1025]
    
    # Get VampNet's intermediate classifier output for proper comparison
    vampnet_x = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_x, "b d n -> b n d")
    vampnet_x = vampnet.transformer(x=vampnet_x, x_mask=torch.ones(1, 10, dtype=torch.bool))
    vampnet_x = rearrange(vampnet_x, "b n d -> b d n")
    vampnet_classifier_out = vampnet.classifier(vampnet_x, None)  # [1, 4096, 10]
    
    print(f"VampNet classifier output: {vampnet_classifier_out.shape}")
    
    # Compare directly at the classifier level
    # VampNet classifier: [1, 4096, 10] where 4096 = 4 codebooks * 1024 vocab
    # V11: [1, 4, 10, 1025] 
    
    print("\nDirect comparison at classifier level:")
    
    total_diff = 0
    total_count = 0
    
    for cb in range(4):
        for pos in range(10):
            # VampNet: get slice for this codebook
            vamp_start = cb * 1024
            vamp_end = (cb + 1) * 1024
            vamp_logits = vampnet_classifier_out[0, vamp_start:vamp_end, pos]
            
            # V11: get logits for this codebook and position
            v11_logits = v11_out[0, cb, pos, :1024]
            
            # Compare
            diff = (vamp_logits - v11_logits).abs()
            total_diff += diff.sum().item()
            total_count += diff.numel()
            
            if cb == 0 and pos < 3:  # Show first few
                print(f"\nCodebook {cb}, Position {pos}:")
                print(f"  VampNet first 5: {vamp_logits[:5]}")
                print(f"  V11 first 5:     {v11_logits[:5]}")
                print(f"  Max diff: {diff.max():.6f}")
    
    mean_diff = total_diff / total_count
    print(f"\nOverall mean absolute difference: {mean_diff:.6f}")
    
    # Compute correlation on flattened outputs with correct ordering
    vamp_flat = []
    v11_flat = []
    
    for cb in range(4):
        for pos in range(10):
            vamp_start = cb * 1024
            vamp_end = (cb + 1) * 1024
            vamp_vec = vampnet_classifier_out[0, vamp_start:vamp_end, pos]
            v11_vec = v11_out[0, cb, pos, :1024]
            vamp_flat.append(vamp_vec)
            v11_flat.append(v11_vec)
    
    vamp_flat = torch.cat(vamp_flat)
    v11_flat = torch.cat(v11_flat)
    
    corr = np.corrcoef(vamp_flat.numpy(), v11_flat.numpy())[0, 1]
    print(f"Correlation: {corr:.4f}")
    
    if corr > 0.99:
        print("\n✅ SUCCESS! V11 matches VampNet!")
        
        # Export to ONNX
        print("\nExporting to ONNX...")
        dummy_latents = torch.randn(1, 32, 100)
        
        torch.onnx.export(
            model,
            dummy_latents,
            "vampnet_transformer_v11.onnx",
            input_names=['latents'],
            output_names=['logits'],
            dynamic_axes={
                'latents': {0: 'batch', 2: 'sequence'},
                'logits': {0: 'batch', 2: 'sequence'}
            },
            opset_version=14,
            verbose=False
        )
        print("✓ Exported to vampnet_transformer_v11.onnx")
        
        # Also save the model state for later use
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'n_codebooks': 4,
                'n_conditioning_codebooks': 0,
                'vocab_size': 1024,
                'latent_dim': 8,
                'd_model': 1280,
                'n_heads': 20,
                'n_layers': 20
            }
        }, 'vampnet_transformer_v11.pth')
        print("✓ Saved model to vampnet_transformer_v11.pth")
        
    else:
        print(f"\n❌ Still not matching (correlation: {corr:.4f})")