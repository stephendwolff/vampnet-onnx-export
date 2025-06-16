#!/usr/bin/env python3
"""
Debug C2F embedding structure to understand how to transfer weights.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from torch.nn.utils import remove_weight_norm


def debug_c2f_embedding():
    """Debug the C2F embedding structure."""
    print("="*80)
    print("DEBUGGING C2F EMBEDDING STRUCTURE")
    print("="*80)
    
    # Load C2F model
    print("\n1. Loading C2F model...")
    c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    
    # Remove weight normalization from classifier
    try:
        remove_weight_norm(c2f.classifier.layers[0])
        print("✓ Removed weight normalization from classifier")
    except:
        print("⚠ No weight normalization to remove")
    
    c2f.eval()
    
    # Check embedding structure
    print(f"\n2. Embedding type: {type(c2f.embedding)}")
    print(f"   Embedding attributes: {dir(c2f.embedding)}")
    
    # Print embedding details
    print(f"\n3. Embedding details:")
    print(f"   n_codebooks: {c2f.embedding.n_codebooks}")
    print(f"   vocab_size: {c2f.embedding.vocab_size}")
    print(f"   emb_dim: {c2f.embedding.emb_dim}")
    print(f"   latent_dim: {c2f.embedding.latent_dim}")
    
    # Check the actual structure
    print(f"\n4. Embedding modules:")
    for name, module in c2f.embedding.named_modules():
        if name:
            print(f"   {name}: {type(module)}")
    
    # Check parameters
    print(f"\n5. Embedding parameters:")
    for name, param in c2f.embedding.named_parameters():
        print(f"   {name}: {param.shape}")
    
    # Test forward pass to understand input/output
    print(f"\n6. Testing forward pass...")
    with torch.no_grad():
        # Test with codes
        test_codes = torch.randint(0, 1024, (1, 14, 10))
        print(f"   Input codes shape: {test_codes.shape}")
        
        # Forward through embedding
        embedded = c2f.embedding(test_codes)
        print(f"   Embedded shape: {embedded.shape}")
        
        # Check if it expects latents or codes
        print(f"\n7. Checking embedding internals...")
        print(f"   special_idxs: {c2f.embedding.special_idxs}")
        print(f"   mask_token: {c2f.mask_token}")
    
    print("="*80)


if __name__ == "__main__":
    debug_c2f_embedding()