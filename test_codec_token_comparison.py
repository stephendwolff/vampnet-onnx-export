#!/usr/bin/env python3
"""Compare token outputs between original VampNet and ONNX encoders."""

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
from pathlib import Path

# Import original VampNet
import vampnet
from vampnet.interface import Interface
from audiotools import AudioSignal

# Import ONNX components
from vampnet_onnx.audio_processor import AudioProcessor

def load_audio(audio_path, target_sr=32000):
    """Load and prepare audio."""
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
    
    return audio, target_sr

def compare_encodings(audio_path):
    """Compare token outputs from both encoders."""
    print(f"Loading audio from {audio_path}")
    audio_np, sr = load_audio(audio_path)
    
    # Original VampNet
    print("\n1. Original VampNet Encoding:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interface = Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device=device
    )
    
    # Prepare audio for VampNet using AudioSignal
    # AudioSignal expects (num_channels, num_samples)
    if audio_np.ndim == 1:
        audio_data = audio_np[np.newaxis, :]  # Add channel dimension
    else:
        audio_data = audio_np
    
    sig = AudioSignal(audio_data, sample_rate=sr)
    
    # Encode with original VampNet
    with torch.no_grad():
        # Use interface.encode which returns the encoded tensor directly
        z_vampnet = interface.encode(sig)
        codes_vampnet = z_vampnet  # z_vampnet is already the codes tensor
    
    print(f"VampNet codes shape: {codes_vampnet.shape}")
    print(f"VampNet codes dtype: {codes_vampnet.dtype}")
    print(f"VampNet codes range: [{codes_vampnet.min().item()}, {codes_vampnet.max().item()}]")
    print(f"VampNet unique values per codebook:")
    for i in range(min(3, codes_vampnet.shape[1])):  # Show first 3 codebooks
        unique_vals = torch.unique(codes_vampnet[0, i, :])
        print(f"  Codebook {i}: {len(unique_vals)} unique values, range [{unique_vals.min().item()}, {unique_vals.max().item()}]")
    
    # ONNX Encoding
    print("\n2. ONNX Encoder:")
    encoder_path = Path("scripts/models/vampnet_codec_encoder_fixed.onnx")
    if not encoder_path.exists():
        print(f"ERROR: Encoder not found at {encoder_path}")
        return None, None
    
    # Create ONNX session
    encoder_session = ort.InferenceSession(str(encoder_path))
    
    # Prepare audio for ONNX
    # Convert numpy to torch tensor and add batch/channel dimensions
    audio_torch_onnx = torch.from_numpy(audio_np).float()
    if audio_torch_onnx.ndim == 1:
        audio_torch_onnx = audio_torch_onnx.unsqueeze(0).unsqueeze(0)  # [batch, channel, samples]
    elif audio_torch_onnx.ndim == 2:
        audio_torch_onnx = audio_torch_onnx.unsqueeze(0)  # [batch, channels, samples]
    
    # Process with AudioProcessor
    audio_processor = AudioProcessor(target_sample_rate=sr)
    with torch.no_grad():
        audio_processed = audio_processor(audio_torch_onnx).numpy()
    
    # Run ONNX encoder
    codes_onnx = encoder_session.run(None, {"audio": audio_processed})[0]
    
    print(f"ONNX codes shape: {codes_onnx.shape}")
    print(f"ONNX codes dtype: {codes_onnx.dtype}")
    print(f"ONNX codes range: [{codes_onnx.min()}, {codes_onnx.max()}]")
    print(f"ONNX unique values per codebook:")
    for i in range(min(3, codes_onnx.shape[1])):  # Show first 3 codebooks
        unique_vals = np.unique(codes_onnx[0, i, :])
        print(f"  Codebook {i}: {len(unique_vals)} unique values, range [{unique_vals.min()}, {unique_vals.max()}]")
    
    # Compare tokens
    print("\n3. Token Comparison:")
    
    # Align shapes for comparison
    min_time = min(codes_vampnet.shape[2], codes_onnx.shape[2])
    codes_vampnet_np = codes_vampnet[0, :, :min_time].cpu().numpy()
    codes_onnx_aligned = codes_onnx[0, :, :min_time]
    
    # Check if tokens match
    exact_match = np.array_equal(codes_vampnet_np, codes_onnx_aligned)
    print(f"Exact token match: {exact_match}")
    
    if not exact_match:
        # Calculate differences
        diff = codes_vampnet_np.astype(np.float32) - codes_onnx_aligned.astype(np.float32)
        print(f"Token differences:")
        print(f"  Mean absolute difference: {np.abs(diff).mean():.4f}")
        print(f"  Max absolute difference: {np.abs(diff).max()}")
        print(f"  Percentage of matching tokens: {(diff == 0).sum() / diff.size * 100:.2f}%")
        
        # Per-codebook analysis
        print(f"\nPer-codebook token match rate:")
        for i in range(codes_vampnet_np.shape[0]):
            match_rate = (codes_vampnet_np[i] == codes_onnx_aligned[i]).mean() * 100
            print(f"  Codebook {i}: {match_rate:.2f}% match")
        
        # Show sample tokens
        print(f"\nSample tokens (first 10 from codebook 0):")
        print(f"  VampNet: {codes_vampnet_np[0, :10]}")
        print(f"  ONNX:    {codes_onnx_aligned[0, :10]}")
    
    # Check token distributions
    print("\n4. Token Distribution Analysis:")
    for i in range(min(3, codes_vampnet_np.shape[0])):
        hist_vampnet, _ = np.histogram(codes_vampnet_np[i], bins=50)
        hist_onnx, _ = np.histogram(codes_onnx_aligned[i], bins=50)
        
        # Normalize histograms
        hist_vampnet = hist_vampnet / hist_vampnet.sum()
        hist_onnx = hist_onnx / hist_onnx.sum()
        
        # Calculate distribution similarity
        kl_div = np.sum(hist_vampnet * np.log(hist_vampnet / (hist_onnx + 1e-8) + 1e-8))
        print(f"  Codebook {i} KL divergence: {kl_div:.4f}")
    
    return codes_vampnet_np, codes_onnx_aligned

def test_transformer_with_different_tokens(codes_vampnet, codes_onnx):
    """Test how transformer behaves with different token inputs."""
    print("\n5. Testing Transformer with Different Tokens:")
    
    # Load transformer
    transformer_path = Path("onnx_models_fixed/coarse_transformer_v2_weighted.onnx")
    if not transformer_path.exists():
        print(f"ERROR: Transformer not found at {transformer_path}")
        return
    
    session = ort.InferenceSession(str(transformer_path))
    
    # Prepare inputs
    batch_size = 1
    seq_len = 100  # Fixed sequence length
    
    # Truncate or pad to match expected length
    if codes_vampnet.shape[1] > seq_len:
        codes_vampnet = codes_vampnet[:, :seq_len]
        codes_onnx = codes_onnx[:, :seq_len]
    elif codes_vampnet.shape[1] < seq_len:
        pad_len = seq_len - codes_vampnet.shape[1]
        codes_vampnet = np.pad(codes_vampnet, ((0, 0), (0, pad_len)), constant_values=0)
        codes_onnx = np.pad(codes_onnx, ((0, 0), (0, pad_len)), constant_values=0)
    
    # Create mask (mask middle portion)
    mask = np.ones((batch_size, seq_len), dtype=np.int64)
    mask[:, 30:70] = 0  # Mask middle 40 tokens
    
    # Test with VampNet tokens
    print("\n  Testing with VampNet tokens:")
    codes_flat_vampnet = codes_vampnet.T.reshape(1, -1).astype(np.int64)
    
    output_vampnet = session.run(None, {
        "codes": codes_flat_vampnet,
        "mask": mask
    })[0]
    
    print(f"  Output shape: {output_vampnet.shape}")
    print(f"  Output range: [{output_vampnet.min()}, {output_vampnet.max()}]")
    
    # Test with ONNX tokens
    print("\n  Testing with ONNX tokens:")
    codes_flat_onnx = codes_onnx.T.reshape(1, -1).astype(np.int64)
    
    output_onnx = session.run(None, {
        "codes": codes_flat_onnx,
        "mask": mask
    })[0]
    
    print(f"  Output shape: {output_onnx.shape}")
    print(f"  Output range: [{output_onnx.min()}, {output_onnx.max()}]")
    
    # Compare outputs
    print("\n  Comparing transformer outputs:")
    diff = np.abs(output_vampnet - output_onnx).mean()
    print(f"  Mean absolute difference in logits: {diff:.4f}")
    
    # Check predicted tokens
    pred_vampnet = np.argmax(output_vampnet[0], axis=-1)
    pred_onnx = np.argmax(output_onnx[0], axis=-1)
    
    match_rate = (pred_vampnet == pred_onnx).mean() * 100
    print(f"  Predicted token match rate: {match_rate:.2f}%")

if __name__ == "__main__":
    # Use example audio
    audio_path = Path("assets/example.wav")
    if not audio_path.exists():
        audio_path = Path("assets/stargazing.wav")
    
    if audio_path.exists():
        codes_vampnet, codes_onnx = compare_encodings(audio_path)
        
        if codes_vampnet is not None and codes_onnx is not None:
            test_transformer_with_different_tokens(codes_vampnet, codes_onnx)
    else:
        print(f"ERROR: No audio file found. Please provide an audio file.")