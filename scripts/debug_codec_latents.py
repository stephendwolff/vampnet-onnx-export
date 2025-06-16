#!/usr/bin/env python3
"""
Debug codec latent conversion to understand the correct shape.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lac.model.lac import LAC as DAC


def debug_codec_latents():
    """Debug how codec converts codes to latents."""
    print("="*80)
    print("DEBUGGING CODEC LATENT CONVERSION")
    print("="*80)
    
    # Load codec
    print("\n1. Loading codec...")
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    # Create test codes
    batch_size = 1
    n_codebooks = 14
    seq_len = 10
    codes = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len))
    print(f"\n2. Input codes shape: {codes.shape}")
    
    # Convert to latents
    print("\n3. Converting codes to latents...")
    with torch.no_grad():
        z = codec.quantizer.from_codes(codes)
        print(f"   Type of z: {type(z)}")
        
        if isinstance(z, tuple):
            print(f"   z is a tuple with {len(z)} elements")
            for i, elem in enumerate(z):
                print(f"   z[{i}] shape: {elem.shape}")
            z = z[0]  # Get the first element
        else:
            print(f"   z shape: {z.shape}")
        
        # Check dimensions
        print(f"\n4. Latent details:")
        print(f"   Shape: {z.shape}")
        print(f"   Expected n_codebooks * latent_dim = 14 * 8 = 112")
        
        # Try different reshaping
        print(f"\n5. Testing reshaping:")
        
        # Option 1: Direct reshape
        try:
            z_reshaped1 = z.view(batch_size, -1, seq_len)
            print(f"   view(batch, -1, seq) -> {z_reshaped1.shape}")
        except:
            print("   view(batch, -1, seq) failed")
        
        # Option 2: Permute then reshape
        try:
            z_permuted = z.permute(0, 1, 3, 2)  # [batch, n_cb, latent_dim, seq]
            z_reshaped2 = z_permuted.reshape(batch_size, -1, seq_len)
            print(f"   permute + reshape -> {z_reshaped2.shape}")
        except Exception as e:
            print(f"   permute + reshape failed: {e}")
        
        # Option 3: Direct computation
        if z.ndim == 4:  # [batch, n_codebooks, seq_len, latent_dim]
            latent_dim = z.shape[-1]
            z_reshaped3 = z.transpose(2, 3).reshape(batch_size, n_codebooks * latent_dim, seq_len)
            print(f"   transpose + reshape -> {z_reshaped3.shape}")
            print(f"   Latent dim detected: {latent_dim}")
    
    print("="*80)


if __name__ == "__main__":
    debug_codec_latents()