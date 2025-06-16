#!/usr/bin/env python3
"""
Final debug of V10 - trace through execution step by step.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10, transfer_weights_v10

# Deterministic
torch.manual_seed(42)
np.random.seed(42)

print("Final V10 debugging...")

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Simple test case
codes = torch.randint(0, 1024, (1, 4, 10))
masked_codes = codes.clone()
masked_codes[:, :, 5:] = 1024

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Run both models
    print("\n1. Running full forward pass...")
    vampnet_out = vampnet(latents)
    
    # Create V10 and run
    model = VampNetTransformerV10(n_layers=20)
    transfer_weights_v10("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
    model.eval()
    
    v10_out = model(latents)
    
    # Compare
    vampnet_reshaped = vampnet_out.reshape(1, 4, 10, 1024)
    v10_truncated = v10_out[:, :, :, :1024]
    
    corr = np.corrcoef(vampnet_reshaped.flatten().numpy(), v10_truncated.flatten().numpy())[0,1]
    print(f"\nCorrelation: {corr:.4f}")
    
    # If still low, let's check what's different
    if corr < 0.9:
        print("\n2. Checking what's different...")
        
        # Are the models in the same mode?
        print(f"VampNet training mode: {vampnet.training}")
        print(f"V10 training mode: {model.training}")
        
        # Check dropout
        print(f"\nVampNet dropout: {vampnet.dropout}")
        
        # Check if there's something we're missing in forward
        print("\n3. Checking VampNet forward implementation...")
        
        # Get VampNet source
        import inspect
        forward_source = inspect.getsource(vampnet.forward)
        
        # Look for any special handling
        if "ctrl" in forward_source:
            print("VampNet uses control conditioning!")
        if "return_activations" in forward_source:
            print("VampNet can return activations")
        
        # Check the mask creation
        print("\n4. Checking mask creation in VampNet...")
        from einops import rearrange
        vampnet_emb = vampnet.embedding(latents)
        x_mask = torch.ones_like(vampnet_emb, dtype=torch.bool)[:, :1, :].squeeze(1)
        print(f"x_mask shape: {x_mask.shape}")
        print(f"x_mask values: {x_mask}")
        
        # Try with explicit parameters
        print("\n5. Running VampNet with explicit parameters...")
        vampnet_out_explicit = vampnet(latents, ctrls=None, ctrl_masks=None, return_activations=False)
        print(f"Outputs match: {torch.allclose(vampnet_out, vampnet_out_explicit)}")
        
        # Check if the issue is in the final rearrange
        print("\n6. Checking final rearrange...")
        print(f"VampNet n_predict_codebooks: {vampnet.n_predict_codebooks}")
        print(f"V10 n_codebooks: {model.n_codebooks}")
        print(f"V10 n_conditioning_codebooks: {model.n_conditioning_codebooks}")
        
        # Manual trace through VampNet
        print("\n7. Manual trace through VampNet...")
        x = vampnet.embedding(latents)
        print(f"After embedding: {x.shape}")
        
        x = rearrange(x, "b d n -> b n d")
        print(f"After rearrange: {x.shape}")
        
        x_mask = torch.ones(1, 10, dtype=torch.bool)
        out = vampnet.transformer(x=x, x_mask=x_mask)
        print(f"After transformer: {out.shape}")
        
        out = rearrange(out, "b n d -> b d n")
        print(f"After rearrange back: {out.shape}")
        
        out = vampnet.classifier(out, None)
        print(f"After classifier: {out.shape}")
        
        out_final = rearrange(out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
        print(f"Final output: {out_final.shape}")
        
        # Check the issue with the final reshape
        print(f"\n8. Understanding the reshape...")
        print(f"Classifier output has {out.shape[1]} channels")
        print(f"n_predict_codebooks = {vampnet.n_predict_codebooks}")
        print(f"So each codebook gets {out.shape[1] // vampnet.n_predict_codebooks} channels")
        print(f"That's {1024} vocab size per codebook")
        
        # The issue might be in how we're comparing
        print("\n9. Correct comparison...")
        # VampNet output is [1, 1024, 40] which reshapes to [1, 4, 10, 1024]
        # But wait - 1024 * 40 = 40960, and 4 * 10 * 1024 = 40960
        # So the reshape is: [batch, vocab*n_codebooks, seq_len] -> [batch, n_codebooks, seq_len, vocab]
        
        vampnet_correct_reshape = rearrange(vampnet_out, "b (v c) (s t) -> b c s (t v)", 
                                           c=4, v=1024, t=4, s=10)
        print(f"Correct reshape attempt: {vampnet_correct_reshape.shape}")
        
    else:
        print("\n✅ Models match!")