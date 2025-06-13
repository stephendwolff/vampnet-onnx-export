#!/usr/bin/env python3
"""Test to understand preprocessing differences between VampNet and ONNX."""

import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from audiotools import AudioSignal
import vampnet
from vampnet_onnx.audio_processor import AudioProcessor

def test_preprocessing():
    """Compare preprocessing steps between VampNet and ONNX."""
    
    # Load test audio
    audio_path = Path("assets/example.wav")
    audio_np, sr = sf.read(audio_path)
    
    print(f"Original audio:")
    print(f"  Shape: {audio_np.shape}")
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(audio_np) / sr:.2f}s")
    
    # 1. VampNet preprocessing
    print(f"\n1. VampNet preprocessing:")
    
    # Create AudioSignal (VampNet's expected input)
    if audio_np.ndim == 1:
        audio_data = audio_np[np.newaxis, :]
    else:
        audio_data = audio_np
        
    sig = AudioSignal(audio_data, sample_rate=sr)
    print(f"  AudioSignal shape: {sig.audio_data.shape}")
    print(f"  AudioSignal sample_rate: {sig.sample_rate}")
    
    # Load VampNet interface
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device=device
    )
    
    # Check codec properties
    print(f"\n  Codec properties:")
    print(f"    Sample rate: {interface.codec.sample_rate}")
    print(f"    Hop length: {interface.codec.hop_length}")
    print(f"    Expected sequence length: {len(audio_np) // interface.codec.hop_length}")
    
    # Preprocess with VampNet
    # VampNet's interface._preprocess does resampling and normalization
    sig_processed = interface._preprocess(sig)
    print(f"\n  After VampNet preprocess:")
    print(f"    Shape: {sig_processed.audio_data.shape}")
    print(f"    Sample rate: {sig_processed.sample_rate}")
    print(f"    Min/Max: [{sig_processed.audio_data.min():.3f}, {sig_processed.audio_data.max():.3f}]")
    
    # 2. ONNX preprocessing
    print(f"\n2. ONNX preprocessing:")
    
    # Use AudioProcessor
    audio_torch = torch.from_numpy(audio_np).float()
    if audio_torch.ndim == 1:
        audio_torch = audio_torch.unsqueeze(0).unsqueeze(0)
    
    processor = AudioProcessor(target_sample_rate=44100)  # VampNet uses 44100
    with torch.no_grad():
        audio_onnx = processor(audio_torch)
    
    print(f"  After ONNX AudioProcessor:")
    print(f"    Shape: {audio_onnx.shape}")
    print(f"    Min/Max: [{audio_onnx.min():.3f}, {audio_onnx.max():.3f}]")
    
    # 3. Compare sequence lengths after encoding
    print(f"\n3. Encoding comparison:")
    
    # VampNet encode
    with torch.no_grad():
        z_vampnet = interface.encode(sig_processed)
    print(f"  VampNet encoded shape: {z_vampnet.shape}")
    
    # Calculate what ONNX would produce
    onnx_seq_len = audio_onnx.shape[-1] // interface.codec.hop_length
    print(f"  ONNX would produce sequence length: {onnx_seq_len}")
    
    # 4. Check sample rate conversion
    print(f"\n4. Sample rate analysis:")
    print(f"  Original sample rate: {sr}")
    print(f"  VampNet expects: {interface.codec.sample_rate}")
    print(f"  Ratio: {sr / interface.codec.sample_rate:.3f}")
    
    # If there's a mismatch, that explains the different sequence lengths
    if sr != interface.codec.sample_rate:
        print(f"\n  ⚠️ Sample rate mismatch! This could explain different sequence lengths.")
        expected_resampled_len = int(len(audio_np) * interface.codec.sample_rate / sr)
        print(f"  After resampling to {interface.codec.sample_rate}Hz: {expected_resampled_len} samples")
        print(f"  Expected sequence length: {expected_resampled_len // interface.codec.hop_length}")

if __name__ == "__main__":
    test_preprocessing()