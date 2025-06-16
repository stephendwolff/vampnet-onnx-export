#!/usr/bin/env python3
"""
Debug output ordering issue.
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

print("Debugging output ordering...")

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
    
    print(f"VampNet output shape: {vampnet_out.shape}")  # [1, 1024, 40]
    print(f"V11 output shape: {v11_out.shape}")          # [1, 4, 10, 1025]
    
    # Let's understand VampNet's output structure
    print("\n1. Understanding VampNet output:")
    print(f"   Shape [1, 1024, 40] means [batch, vocab_size, seq_len * n_codebooks]")
    print(f"   n_predict_codebooks = {vampnet.n_predict_codebooks}")
    
    # The rearrange pattern in VampNet is: "b (p c) t -> b p (t c)"
    # This means the classifier output [1, 4096, 10] is rearranged as:
    # p = vocab_size = 1024
    # c = n_predict_codebooks = 4
    # t = seq_len = 10
    
    print("\n2. Checking the rearrange pattern:")
    
    # Let's trace through VampNet's output manually
    # Simulate classifier output
    test_classifier_out = torch.arange(4096 * 10).float().reshape(1, 4096, 10)
    
    # Apply VampNet's rearrange
    test_rearranged = rearrange(test_classifier_out, "b (p c) t -> b p (t c)", c=4)
    print(f"   Test rearranged shape: {test_rearranged.shape}")
    
    # Check first few values
    print(f"\n   Classifier [0, 0:4, 0]: {test_classifier_out[0, 0:4, 0]}")
    print(f"   Rearranged [0, 0, 0:4]: {test_rearranged[0, 0, 0:4]}")
    
    # This shows how the pattern works:
    # classifier[0, 0, 0] -> rearranged[0, 0, 0]
    # classifier[0, 1024, 0] -> rearranged[0, 0, 1]
    # classifier[0, 2048, 0] -> rearranged[0, 0, 2]
    # classifier[0, 3072, 0] -> rearranged[0, 0, 3]
    
    print("\n3. Checking actual outputs:")
    
    # Get VampNet classifier output before rearrange
    from einops import rearrange as rearrange_orig
    vampnet_x = vampnet.embedding(latents)
    vampnet_x = rearrange_orig(vampnet_x, "b d n -> b n d")
    vampnet_x = vampnet.transformer(x=vampnet_x, x_mask=torch.ones(1, 10, dtype=torch.bool))
    vampnet_x = rearrange_orig(vampnet_x, "b n d -> b d n")
    vampnet_classifier_out = vampnet.classifier(vampnet_x, None)
    
    print(f"\n   VampNet classifier output shape: {vampnet_classifier_out.shape}")
    print(f"   First few values at position 0:")
    for i in range(4):
        idx = i * 1024
        print(f"   Codebook {i} (idx {idx}): {vampnet_classifier_out[0, idx, 0]:.4f}")
    
    # Now check what V11 produces
    print(f"\n   V11 output at position 0:")
    for i in range(4):
        print(f"   Codebook {i}: {v11_out[0, i, 0, 0]:.4f}")
    
    # The issue might be in how we're comparing
    print("\n4. Correct comparison:")
    
    # VampNet: "b (p c) t -> b p (t c)" where p=vocab, c=codebooks, t=seq
    # This gives [batch, vocab, seq*codebooks]
    
    # To match V11 [batch, codebooks, seq, vocab], we need:
    # "b (v c) (s cb) -> b cb s v" where v=vocab, c=1, s=seq, cb=codebooks
    
    # Actually, let's think about this differently
    # VampNet classifier output is [1, 4096, 10]
    # It's organized as [batch, codebook0_vocab + codebook1_vocab + ..., seq]
    
    # After rearrange "b (p c) t -> b p (t c)" with p=1024, c=4:
    # Output is [1, 1024, 40]
    # This means for each vocab index, we have all positions across all codebooks
    
    print("\n5. Figuring out the correct mapping:")
    
    # Create a simple test to understand the mapping
    test_in = torch.arange(4096).float().reshape(1, 4096, 1)
    test_out = rearrange(test_in, "b (p c) t -> b p (t c)", p=1024, c=4)
    print(f"   Test mapping shape: {test_out.shape}")
    
    # Check where each codebook's first vocab item ends up
    for cb in range(4):
        original_idx = cb * 1024
        print(f"   Codebook {cb} vocab 0 (idx {original_idx}): maps to position {test_out[0, 0, cb]}")