#!/usr/bin/env python3
"""Test the pre-padded ONNX encoder to verify it produces correct tokens."""

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from audiotools import AudioSignal
import vampnet

def pad_audio(audio, hop_length=768):
    """Pad audio to multiple of hop_length."""
    if audio.ndim == 1:
        audio = audio[np.newaxis, np.newaxis, :]
    elif audio.ndim == 2:
        audio = audio[np.newaxis, :]
        
    batch, channels, samples = audio.shape
    padded_samples = ((samples + hop_length - 1) // hop_length) * hop_length
    pad_amount = padded_samples - samples
    
    if pad_amount > 0:
        audio = np.pad(audio, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
    
    return audio, samples

def test_prepadded_encoder():
    """Test the pre-padded encoder with real audio."""
    
    # Load test audio
    audio_path = Path("assets/example.wav")
    audio_np, sr = sf.read(audio_path)
    
    print(f"Test audio: {len(audio_np)} samples at {sr}Hz")
    
    # Load VampNet for comparison
    device = 'cpu'
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device=device
    )
    
    # 1. Encode with VampNet
    print("\n1. VampNet encoding:")
    sig = AudioSignal(audio_np[np.newaxis, :], sample_rate=sr)
    z_vampnet = interface.encode(sig)
    print(f"  Shape: {z_vampnet.shape}")
    print(f"  Sample tokens: {z_vampnet[0, 0, :10].cpu().numpy()}")
    
    # 2. Encode with ONNX pre-padded encoder
    print("\n2. ONNX pre-padded encoder:")
    
    encoder_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    if not encoder_path.exists():
        print(f"ERROR: Encoder not found at {encoder_path}")
        return
    
    session = ort.InferenceSession(str(encoder_path))
    
    # Pad the audio
    audio_padded, original_length = pad_audio(audio_np.astype(np.float32))
    print(f"  Padded shape: {audio_padded.shape}")
    
    # Encode
    codes_onnx = session.run(None, {'audio_padded': audio_padded})[0]
    print(f"  Output shape: {codes_onnx.shape}")
    print(f"  Sample tokens: {codes_onnx[0, 0, :10]}")
    
    # 3. Compare tokens
    print("\n3. Token comparison:")
    
    # Align lengths for comparison
    min_len = min(z_vampnet.shape[2], codes_onnx.shape[2])
    
    # Compare first few tokens
    vampnet_tokens = z_vampnet[0, :, :min_len].cpu().numpy()
    onnx_tokens = codes_onnx[0, :, :min_len]
    
    # Exact match check
    exact_match = np.array_equal(vampnet_tokens, onnx_tokens)
    print(f"  Exact match: {exact_match}")
    
    if not exact_match:
        # Calculate match rate
        matches = (vampnet_tokens == onnx_tokens)
        match_rate = matches.mean() * 100
        print(f"  Token match rate: {match_rate:.1f}%")
        
        # Per-codebook analysis
        print(f"\n  Per-codebook match rates:")
        for i in range(min(5, vampnet_tokens.shape[0])):
            cb_match_rate = matches[i].mean() * 100
            print(f"    Codebook {i}: {cb_match_rate:.1f}%")
        
        # Show differences in first codebook
        print(f"\n  First 20 tokens comparison (Codebook 0):")
        print(f"    VampNet: {vampnet_tokens[0, :20]}")
        print(f"    ONNX:    {onnx_tokens[0, :20]}")
    else:
        print("  ✓ Tokens match perfectly!")
    
    # Test with different lengths
    print("\n4. Testing different audio lengths:")
    
    test_lengths = [1, 2, 5, 10]
    all_correct = True
    
    for seconds in test_lengths:
        samples = int(seconds * sr)
        test_audio = audio_np[:samples] if samples < len(audio_np) else np.tile(audio_np, 2)[:samples]
        
        # Pad and encode
        test_padded, _ = pad_audio(test_audio.astype(np.float32))
        codes = session.run(None, {'audio_padded': test_padded})[0]
        
        expected_tokens = test_padded.shape[2] // 768
        is_correct = codes.shape[2] == expected_tokens
        status = "✓" if is_correct else "✗"
        all_correct &= is_correct
        
        print(f"  {seconds}s: {codes.shape[2]} tokens (expected {expected_tokens}) {status}")
    
    if all_correct:
        print("\n🎉 Pre-padded encoder works correctly for all lengths!")
    
    return exact_match

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    match = test_prepadded_encoder()