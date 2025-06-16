#!/usr/bin/env python3
"""
Simple working demo that follows VampNet's interface exactly.
Based on step8_end_to_end_comparison.py but simplified and corrected.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
import soundfile as sf

sys.path.append(str(Path(__file__).parent))

from vampnet.interface import Interface
from vampnet.mask import linear_random, codebook_mask
import audiotools as at
import onnxruntime as ort

print("="*80)
print("SIMPLE VAMPNET ONNX DEMO")
print("="*80)

# 1. Load VampNet interface
print("\n1. Loading VampNet Interface...")
interface = Interface(
    coarse_ckpt="models/vampnet/coarse.pth",
    coarse2fine_ckpt="models/vampnet/c2f.pth",
    codec_ckpt="models/vampnet/codec.pth",
    device="cpu",
    wavebeat_ckpt=None,  # Skip beat tracking
    compile=False
)
print("✓ VampNet loaded")

# 2. Load ONNX models
print("\n2. Loading ONNX models...")

# Load encoder
encoder_path = "scripts/models/vampnet_encoder_prepadded.onnx"
if Path(encoder_path).exists():
    encoder_session = ort.InferenceSession(encoder_path)
    print(f"✓ Encoder loaded from {encoder_path}")
else:
    print(f"❌ Encoder not found at {encoder_path}")
    encoder_session = None

# Load coarse transformer
coarse_path = "vampnet_transformer_v11.onnx"
if Path(coarse_path).exists():
    coarse_session = ort.InferenceSession(coarse_path)
    print(f"✓ Coarse transformer loaded from {coarse_path}")
else:
    print(f"❌ Coarse transformer not found at {coarse_path}")
    coarse_session = None

# Load decoder
decoder_path = "scripts/models/vampnet_codec_decoder.onnx"
if Path(decoder_path).exists():
    decoder_session = ort.InferenceSession(decoder_path)
    print(f"✓ Decoder loaded from {decoder_path}")
else:
    print(f"❌ Decoder not found at {decoder_path}")
    decoder_session = None

# 3. Create test audio (100 tokens worth)
print("\n3. Creating test audio...")
sample_rate = 44100
hop_length = 768
n_tokens = 100
target_samples = n_tokens * hop_length  # 76800 samples
duration = target_samples / sample_rate

# Create simple test tone
t = np.linspace(0, duration, target_samples)
test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
test_audio = test_audio.astype(np.float32)

# Create AudioSignal
test_signal = at.AudioSignal(test_audio[None, :], sample_rate)
print(f"✓ Created {duration:.2f}s test audio ({n_tokens} tokens)")

# Save input
sf.write('simple_demo_input.wav', test_audio, sample_rate)

# 4. VampNet processing
print("\n4. Running VampNet...")
with torch.no_grad():
    # a) Encode
    print("  a) Encoding...")
    z = interface.encode(test_signal)
    print(f"     Encoded shape: {z.shape}")
    
    # b) Create mask (following VampNet exactly)
    print("  b) Creating mask...")
    mask = interface.build_mask(
        z,
        test_signal,
        rand_mask_intensity=0.8,
        upper_codebook_mask=3
    )
    print(f"     Mask shape: {mask.shape}")
    print(f"     Masked positions: {mask.sum().item()}")
    
    # c) Run vamp (VampNet's complete pipeline)
    print("  c) Running vamp...")
    vampnet_start = time.time()
    
    z_vamped = interface.vamp(
        z,
        mask=mask,
        temperature=1.0,
        top_p=0.9,
        return_mask=False
    )
    
    vampnet_time = time.time() - vampnet_start
    print(f"     Output shape: {z_vamped.shape}")
    print(f"     Time: {vampnet_time:.2f}s")
    
    # d) Decode
    print("  d) Decoding...")
    audio_vampnet = interface.decode(z_vamped)
    audio_vampnet_np = audio_vampnet.audio_data.squeeze().cpu().numpy()
    
# Save VampNet output
sf.write('simple_demo_vampnet.wav', audio_vampnet_np, sample_rate)
print(f"✓ Saved VampNet output")

# 5. ONNX processing (following VampNet's steps)
print("\n5. Running ONNX...")

if encoder_session and coarse_session and decoder_session:
    # a) Encode with ONNX
    print("  a) Encoding with ONNX...")
    # Pad audio for encoder
    audio_padded = test_audio[np.newaxis, np.newaxis, :]
    codes_onnx = encoder_session.run(None, {'audio_padded': audio_padded})[0]
    codes_onnx_torch = torch.from_numpy(codes_onnx).long()
    print(f"     Encoded shape: {codes_onnx.shape}")
    
    # b) Use same mask as VampNet
    print("  b) Using VampNet mask...")
    
    # c) ONNX generation (simplified single pass for now)
    print("  c) Running ONNX transformer...")
    onnx_start = time.time()
    
    # Load the iterative generator for proper generation
    from scripts.iterative_generation import create_onnx_generator
    
    # Create coarse generator
    coarse_generator = create_onnx_generator(
        coarse_path,
        "models/vampnet/codec.pth",
        n_codebooks=4,
        latent_dim=8,
        mask_token=1024
    )
    
    # Generate with coarse model
    coarse_codes = codes_onnx_torch[:, :4, :]
    coarse_mask = mask[:, :4, :]
    
    z_generated = coarse_generator.generate(
        start_tokens=coarse_codes,
        mask=coarse_mask,
        temperature=1.0,
        top_p=0.9,
        time_steps=12  # Match VampNet's default
    )
    
    onnx_time = time.time() - onnx_start
    print(f"     Generated shape: {z_generated.shape}")
    print(f"     Time: {onnx_time:.2f}s")
    
    # d) Decode with ONNX
    print("  d) Decoding with ONNX...")
    # Pad to 14 codebooks for decoder
    codes_full = np.zeros((1, 14, z_generated.shape[2]), dtype=np.int64)
    codes_full[:, :4, :] = z_generated.numpy()
    
    audio_onnx = decoder_session.run(None, {'codes': codes_full})[0]
    audio_onnx_np = audio_onnx.squeeze()
    
    # Save ONNX output
    sf.write('simple_demo_onnx.wav', audio_onnx_np, sample_rate)
    print(f"✓ Saved ONNX output")
    
    # 6. Compare
    print("\n6. Comparison:")
    print(f"  VampNet time: {vampnet_time:.2f}s")
    print(f"  ONNX time: {onnx_time:.2f}s")
    print(f"  Speedup: {vampnet_time/onnx_time:.1f}x")
    
    # Audio comparison
    min_len = min(len(audio_vampnet_np), len(audio_onnx_np))
    mse = np.mean((audio_vampnet_np[:min_len] - audio_onnx_np[:min_len])**2)
    print(f"  MSE: {mse:.6f}")
    
else:
    print("❌ Missing required ONNX models")

print("\n✅ Demo complete!")
print("Generated files:")
print("  - simple_demo_input.wav (input audio)")
print("  - simple_demo_vampnet.wav (VampNet output)")
print("  - simple_demo_onnx.wav (ONNX output)")