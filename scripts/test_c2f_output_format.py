#!/usr/bin/env python3
"""
Test C2F output format to understand the correct rearrangement.
"""

import torch
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm


def test_c2f_output_format():
    """Test C2F output format and rearrangement."""
    print("="*80)
    print("TESTING C2F OUTPUT FORMAT")
    print("="*80)
    
    # Load C2F
    print("\n1. Loading C2F model...")
    c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    try:
        remove_weight_norm(c2f.classifier.layers[0])
    except:
        pass
    c2f.eval()
    
    print(f"   n_predict_codebooks: {c2f.n_predict_codebooks}")
    print(f"   classifier out_channels: {c2f.classifier.layers[0].out_channels}")
    
    # Load codec
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    # Test with different sequence lengths
    print("\n2. Testing with different sequence lengths...")
    for seq_len in [10, 20]:
        print(f"\n   Sequence length: {seq_len}")
        
        # Create input
        codes = torch.randint(0, 1024, (1, 14, seq_len))
        z = codec.quantizer.from_codes(codes)
        latents = z[1]
        
        # Forward pass
        with torch.no_grad():
            # Direct forward pass
            output = c2f(latents)
            print(f"   Final output shape: {output.shape}")
            
            # The output shape tells us about the rearrangement
            # If output is [1, 1024, seq_len * 10], then:
            # - 1024 is the vocab_size (p in the rearrange)
            # - seq_len * 10 is the flattened sequence dimension
            
            # Verify the math
            expected_flat_seq = seq_len * c2f.n_predict_codebooks
            actual_flat_seq = output.shape[2]
            print(f"   Expected flattened seq: {expected_flat_seq}")
            print(f"   Actual flattened seq: {actual_flat_seq}")
            print(f"   Match: {expected_flat_seq == actual_flat_seq}")
    
    # Test the generate method
    print("\n3. Testing generate method output...")
    with torch.no_grad():
        mask = torch.zeros_like(codes)
        mask[:, 4:, :] = 1  # Mask non-conditioning codebooks
        
        generated = c2f.generate(
            codec=codec,
            time_steps=seq_len,
            start_tokens=codes,
            mask=mask,
            return_signal=False,
            _sampling_steps=1
        )
        print(f"   Generated shape: {generated.shape}")
    
    print("="*80)


if __name__ == "__main__":
    test_c2f_output_format()