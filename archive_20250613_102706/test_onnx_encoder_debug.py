#!/usr/bin/env python3
"""Debug why ONNX encoder produces different tokens and sequence length."""

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from audiotools import AudioSignal
import vampnet
from vampnet_onnx.audio_processor import AudioProcessor

def debug_onnx_encoder():
    """Debug ONNX encoder issues."""
    
    # Load test audio
    audio_path = Path("assets/example.wav")
    audio_np, sr = sf.read(audio_path)
    
    print(f"Test audio: {len(audio_np)} samples at {sr}Hz = {len(audio_np)/sr:.2f}s")
    
    # Load VampNet for comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device=device
    )
    
    print(f"\nCodec info:")
    print(f"  Hop length: {interface.codec.hop_length}")
    print(f"  Expected tokens: {len(audio_np) // interface.codec.hop_length}")
    
    # 1. Test with VampNet directly
    print(f"\n1. VampNet encoding:")
    sig = AudioSignal(audio_np[np.newaxis, :], sample_rate=sr)
    sig_processed = interface._preprocess(sig)
    
    with torch.no_grad():
        z_vampnet = interface.encode(sig_processed)
    print(f"  Shape: {z_vampnet.shape}")
    
    # 2. Test ONNX encoder with different input shapes
    print(f"\n2. ONNX encoder tests:")
    
    encoder_path = Path("scripts/models/vampnet_codec_encoder.onnx")
    if not encoder_path.exists():
        print(f"ERROR: Encoder not found at {encoder_path}")
        return
        
    # Load ONNX model and check its inputs
    encoder_session = ort.InferenceSession(str(encoder_path))
    
    print(f"\n  ONNX model info:")
    for inp in encoder_session.get_inputs():
        print(f"    Input '{inp.name}': shape={inp.shape}, dtype={inp.type}")
    for out in encoder_session.get_outputs():
        print(f"    Output '{out.name}': shape={out.shape}, dtype={out.type}")
    
    # Test 1: Raw numpy audio
    print(f"\n  Test 1: Raw numpy input")
    try:
        audio_input = audio_np[np.newaxis, np.newaxis, :].astype(np.float32)
        print(f"    Input shape: {audio_input.shape}")
        codes = encoder_session.run(None, {"audio": audio_input})[0]
        print(f"    Output shape: {codes.shape}")
        print(f"    Sequence length: {codes.shape[2]}")
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # Test 2: With AudioProcessor
    print(f"\n  Test 2: With AudioProcessor")
    try:
        audio_torch = torch.from_numpy(audio_np).float()
        if audio_torch.ndim == 1:
            audio_torch = audio_torch.unsqueeze(0).unsqueeze(0)
        
        processor = AudioProcessor(target_sample_rate=sr)
        with torch.no_grad():
            audio_processed = processor(audio_torch).numpy()
        
        print(f"    Processed shape: {audio_processed.shape}")
        codes = encoder_session.run(None, {"audio": audio_processed})[0]
        print(f"    Output shape: {codes.shape}")
        print(f"    Sequence length: {codes.shape[2]}")
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # Test 3: Shorter audio to understand the relationship
    print(f"\n  Test 3: Testing with different audio lengths")
    for test_seconds in [1, 2, 3]:
        test_samples = int(test_seconds * sr)
        test_audio = audio_np[:test_samples]
        test_input = test_audio[np.newaxis, np.newaxis, :].astype(np.float32)
        
        codes = encoder_session.run(None, {"audio": test_input})[0]
        expected_tokens = test_samples // interface.codec.hop_length
        
        print(f"    {test_seconds}s audio ({test_samples} samples):")
        print(f"      Expected tokens: {expected_tokens}")
        print(f"      Actual tokens: {codes.shape[2]}")
        print(f"      Ratio: {codes.shape[2] / expected_tokens:.3f}")
    
    # 3. Check if there's a downsampling factor
    print(f"\n3. Analyzing token length ratio:")
    full_codes = encoder_session.run(None, {"audio": audio_np[np.newaxis, np.newaxis, :].astype(np.float32)})[0]
    actual_tokens = full_codes.shape[2]
    expected_tokens = len(audio_np) // interface.codec.hop_length
    
    ratio = actual_tokens / expected_tokens
    print(f"  Expected tokens: {expected_tokens}")
    print(f"  Actual tokens: {actual_tokens}")
    print(f"  Ratio: {ratio:.3f}")
    
    if abs(ratio - 0.3) < 0.1:
        print(f"\n  ⚠️ Looks like ONNX encoder has ~3x downsampling!")
        print(f"  This could be due to:")
        print(f"    - Wrong hop_length in export (using 768*3 = 2304?)")
        print(f"    - Additional downsampling layer")
        print(f"    - Wrong model architecture")

if __name__ == "__main__":
    debug_onnx_encoder()