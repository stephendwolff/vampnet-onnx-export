"""
Test script to compare the original VampNet codec with ONNX export.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import os

# Import vampnet components
import vampnet
import audiotools as at

# Import our ONNX export components
from vampnet_onnx.vampnet_codec import VampNetCodecEncoder, VampNetCodecDecoder
from vampnet_onnx.audio_processor import AudioProcessor


def load_test_audio(audio_path: str = None, duration: float = 2.0):
    """Load or generate test audio."""
    if audio_path and os.path.exists(audio_path):
        # Load from file
        audio, sr = sf.read(audio_path)
        # Convert to torch tensor
        audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        return audio, sr
    else:
        # Generate test tone
        sr = 44100
        t = torch.linspace(0, duration, int(sr * duration))
        # Create a simple melody
        freq1, freq2, freq3 = 440, 554, 659  # A4, C#5, E5
        audio = 0.3 * (torch.sin(2 * np.pi * freq1 * t) + 
                      torch.sin(2 * np.pi * freq2 * t) * 0.8 +
                      torch.sin(2 * np.pi * freq3 * t) * 0.6)
        audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        return audio, sr


def test_original_vampnet_codec(audio, sr):
    """Test the original VampNet codec."""
    print("\n=== Testing Original VampNet Codec ===")
    
    # Load VampNet interface with local model
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth"
    )
    codec = interface.codec
    
    # Convert to AudioSignal
    signal = at.AudioSignal(audio.numpy(), sr)
    
    # Preprocess (matching original vampnet)
    signal = interface._preprocess(signal)
    print(f"Preprocessed shape: {signal.samples.shape}")
    
    # Encode
    z = codec.encode(signal.samples, signal.sample_rate)
    codes = z["codes"]
    print(f"Encoded codes shape: {codes.shape}")
    
    # Decode
    reconstructed = codec.decode(codes)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    return signal.samples, codes, reconstructed


def test_onnx_codec(audio, sr):
    """Test our ONNX-compatible codec wrapper."""
    print("\n=== Testing ONNX Codec Wrapper ===")
    
    # Load VampNet interface with local model
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth"
    )
    codec = interface.codec
    
    # Create audio processor
    processor = AudioProcessor(
        target_sample_rate=44100,
        target_loudness=-24.0,
        hop_length=codec.hop_length
    )
    
    # Create encoder/decoder wrappers
    encoder = VampNetCodecEncoder(codec)
    decoder = VampNetCodecDecoder(codec)
    
    # Preprocess audio
    processed_audio = processor(audio)
    print(f"Processed audio shape: {processed_audio.shape}")
    
    # Encode
    codes = encoder(processed_audio)
    print(f"Encoded codes shape: {codes.shape}")
    
    # Decode
    reconstructed = decoder(codes)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    return processed_audio, codes, reconstructed


def compute_metrics(original, reconstructed):
    """Compute quality metrics."""
    # Ensure same shape
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]
    
    # MSE
    mse = torch.mean((original - reconstructed) ** 2)
    
    # SNR
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # Max absolute error
    max_error = torch.max(torch.abs(original - reconstructed))
    
    return {
        'mse': mse.item(),
        'snr_db': snr.item(),
        'max_error': max_error.item()
    }


def save_audio(audio, path, sr=44100):
    """Save audio tensor to file."""
    # Convert to numpy and ensure proper shape
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if audio.ndim == 3:
        audio = audio[0]
    
    # Convert to (samples, channels) for soundfile
    if audio.ndim == 2:
        audio = audio.T
    else:
        audio = audio.reshape(-1, 1)
    
    # Ensure in range [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    
    sf.write(path, audio, sr)
    print(f"Saved audio to {path}")


def main():
    # Create output directory
    output_dir = Path("outputs/codec_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate test audio
    print("Loading test audio...")
    audio, sr = load_test_audio(duration=3.0)
    print(f"Test audio shape: {audio.shape}, sample rate: {sr}")
    
    # Save original
    save_audio(audio, output_dir / "original.wav", sr)
    
    # Test original codec
    orig_processed, orig_codes, orig_reconstructed = test_original_vampnet_codec(audio, sr)
    save_audio(orig_reconstructed, output_dir / "vampnet_original.wav", sr)
    
    # Test ONNX codec
    onnx_processed, onnx_codes, onnx_reconstructed = test_onnx_codec(audio, sr)
    save_audio(onnx_reconstructed, output_dir / "vampnet_onnx.wav", sr)
    
    # Compare preprocessing
    print("\n=== Preprocessing Comparison ===")
    print(f"Original preprocessing output shape: {orig_processed.shape}")
    print(f"ONNX preprocessing output shape: {onnx_processed.shape}")
    
    # Ensure tensors for comparison
    if isinstance(orig_processed, np.ndarray):
        orig_processed = torch.from_numpy(orig_processed)
    
    preprocess_metrics = compute_metrics(orig_processed, onnx_processed)
    print(f"Preprocessing MSE: {preprocess_metrics['mse']:.6f}")
    print(f"Preprocessing SNR: {preprocess_metrics['snr_db']:.2f} dB")
    
    # Compare codes
    print("\n=== Codes Comparison ===")
    print(f"Original codes shape: {orig_codes.shape}")
    print(f"ONNX codes shape: {onnx_codes.shape}")
    print(f"Codes match: {torch.allclose(orig_codes, onnx_codes)}")
    
    # Compare reconstructions
    print("\n=== Reconstruction Comparison ===")
    
    # Ensure tensors for comparison
    if isinstance(orig_reconstructed, np.ndarray):
        orig_reconstructed = torch.from_numpy(orig_reconstructed)
    
    metrics = compute_metrics(orig_reconstructed, onnx_reconstructed)
    print(f"Reconstruction MSE: {metrics['mse']:.6f}")
    print(f"Reconstruction SNR: {metrics['snr_db']:.2f} dB")
    print(f"Max absolute error: {metrics['max_error']:.6f}")
    
    # Compare against original audio
    print("\n=== Original vs Reconstructed ===")
    # Need to ensure same shape
    orig_audio = audio.squeeze()
    if orig_audio.dim() == 1:
        orig_audio = orig_audio.unsqueeze(0)
    
    orig_metrics = compute_metrics(orig_audio, orig_reconstructed)
    onnx_metrics = compute_metrics(orig_audio, onnx_reconstructed)
    
    print(f"Original VampNet - MSE: {orig_metrics['mse']:.6f}, SNR: {orig_metrics['snr_db']:.2f} dB")
    print(f"ONNX VampNet - MSE: {onnx_metrics['mse']:.6f}, SNR: {onnx_metrics['snr_db']:.2f} dB")
    
    print(f"\nAll audio files saved to {output_dir}")


if __name__ == "__main__":
    main()