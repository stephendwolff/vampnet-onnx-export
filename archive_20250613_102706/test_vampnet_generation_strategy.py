#!/usr/bin/env python3
"""
Test VampNet's generation strategy to understand how it differs from ONNX.
"""

import torch
import numpy as np
import vampnet
from pathlib import Path
import soundfile as sf


def test_vampnet_generation():
    """Analyze how VampNet generates tokens."""
    
    print("=== Testing VampNet Generation Strategy ===\n")
    
    # Load VampNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth"
    )
    
    # Load test audio
    audio_path = Path("assets/example.wav")
    audio, sr = sf.read(audio_path)
    
    from audiotools import AudioSignal
    sig = AudioSignal(audio[np.newaxis, :], sample_rate=sr)
    
    # Encode
    with torch.no_grad():
        z = interface.encode(sig)
    
    print(f"Original codes shape: {z.shape}")
    print(f"Original codes range: [{z.min().item()}, {z.max().item()}]")
    print(f"Sample codes: {z[0, 0, :10].cpu().numpy()}")
    
    # Create mask
    mask = interface.build_mask(
        z,
        sig,
        periodic_prompt=30,
        upper_codebook_mask=3
    )
    
    print(f"\nMask shape: {mask.shape}")
    print(f"Mask values: {torch.unique(mask)}")
    print(f"Masked positions (mask==0): {(mask == 0).sum().item()}")
    
    # Now let's trace through the generation process
    print("\n=== Tracing Generation Process ===")
    
    # Get the coarse model directly
    coarse_model = interface.coarse
    print(f"\nCoarse model class: {type(coarse_model)}")
    print(f"Mask token: {coarse_model.mask_token}")
    
    # Apply mask to codes
    z_masked = z.clone()
    z_masked[mask == 0] = coarse_model.mask_token
    
    print(f"\nAfter masking:")
    print(f"Masked codes range: [{z_masked.min().item()}, {z_masked.max().item()}]")
    print(f"Number of mask tokens: {(z_masked == coarse_model.mask_token).sum().item()}")
    
    # Check the actual generation method
    print("\n=== Checking Generation Method ===")
    
    # VampNet uses iterative generation
    with torch.no_grad():
        # This is what happens inside vamp()
        z_gen = z_masked.clone()
        
        # Get logits from model
        logits = coarse_model(z_gen)
        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        
        # Sample from logits
        # VampNet samples from logits at masked positions
        probs = torch.softmax(logits / 1.0, dim=1)  # temperature=1.0
        
        # Sample tokens
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
        sampled = sampled.reshape(z_gen.shape)
        
        print(f"\nSampled tokens shape: {sampled.shape}")
        print(f"Sampled range: [{sampled.min().item()}, {sampled.max().item()}]")
        
        # Key insight: VampNet only replaces masked positions
        z_gen[mask == 0] = sampled[mask == 0]
        
        print(f"\nAfter sampling:")
        print(f"Generated range: [{z_gen.min().item()}, {z_gen.max().item()}]")
        print(f"Mask tokens remaining: {(z_gen == coarse_model.mask_token).sum().item()}")
    
    # Compare with full vamp() method
    print("\n=== Full VampNet Generation ===")
    with torch.no_grad():
        z_full, _ = interface.vamp(
            z,
            mask=mask,
            temperature=1.0,
            top_p=0.9,
            return_mask=True
        )
    
    print(f"Full generation shape: {z_full.shape}")
    print(f"Full generation range: [{z_full.min().item()}, {z_full.max().item()}]")
    
    # Check if mask tokens appear in output
    mask_tokens_in_output = (z_full == coarse_model.mask_token).sum().item()
    print(f"Mask tokens in final output: {mask_tokens_in_output}")
    
    # Analyze token distribution
    print(f"\nToken distribution:")
    unique_orig = len(torch.unique(z))
    unique_gen = len(torch.unique(z_full))
    print(f"Original unique tokens: {unique_orig}")
    print(f"Generated unique tokens: {unique_gen}")
    
    return interface, z, z_full, mask


if __name__ == "__main__":
    interface, z_orig, z_gen, mask = test_vampnet_generation()