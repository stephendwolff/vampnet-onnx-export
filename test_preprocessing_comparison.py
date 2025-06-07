"""
Test to compare preprocessing between original VampNet and our ONNX implementation.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Import vampnet components
import vampnet
import audiotools as at

# Import our ONNX export components
from vampnet_onnx.audio_processor import AudioProcessor


def generate_test_audio(duration: float = 2.0):
    """Generate test audio."""
    sr = 44100
    t = torch.linspace(0, duration, int(sr * duration))
    # Create a simple melody with varying amplitude
    freq1, freq2, freq3 = 440, 554, 659  # A4, C#5, E5
    audio = 0.3 * (torch.sin(2 * np.pi * freq1 * t) + 
                   torch.sin(2 * np.pi * freq2 * t) * 0.8 +
                   torch.sin(2 * np.pi * freq3 * t) * 0.6)
    
    # Add some dynamics
    envelope = torch.sin(np.pi * t / duration) ** 2
    audio = audio * envelope
    
    audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    return audio, sr


def test_original_preprocessing(audio, sr):
    """Test original VampNet preprocessing."""
    print("\n=== Original VampNet Preprocessing ===")
    
    # Load interface
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth"
    )
    
    # Convert to AudioSignal
    signal = at.AudioSignal(audio.numpy(), sr)
    print(f"Input shape: {signal.samples.shape}")
    # Convert to numpy for calculations
    input_samples = signal.samples.detach().cpu().numpy() if isinstance(signal.samples, torch.Tensor) else signal.samples
    print(f"Input RMS: {np.sqrt(np.mean(input_samples**2)):.6f}")
    print(f"Input max: {np.max(np.abs(input_samples)):.6f}")
    
    # Apply preprocessing
    processed = interface._preprocess(signal)
    print(f"\nProcessed shape: {processed.samples.shape}")
    # Convert to numpy for calculations
    processed_samples = processed.samples.detach().cpu().numpy() if isinstance(processed.samples, torch.Tensor) else processed.samples
    print(f"Processed RMS: {np.sqrt(np.mean(processed_samples**2)):.6f}")
    print(f"Processed max: {np.max(np.abs(processed_samples)):.6f}")
    
    return processed.samples


def test_onnx_preprocessing(audio, sr):
    """Test ONNX preprocessing."""
    print("\n=== ONNX Preprocessing ===")
    
    # Load codec just to get hop_length
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth"
    )
    codec = interface.codec
    
    # Create processor
    processor = AudioProcessor(
        target_sample_rate=44100,
        target_loudness=-24.0,
        hop_length=codec.hop_length
    )
    
    print(f"Input shape: {audio.shape}")
    print(f"Input RMS: {torch.sqrt(torch.mean(audio**2)).item():.6f}")
    print(f"Input max: {torch.max(torch.abs(audio)).item():.6f}")
    
    # Process
    processed = processor(audio)
    print(f"\nProcessed shape: {processed.shape}")
    print(f"Processed RMS: {torch.sqrt(torch.mean(processed**2)).item():.6f}") 
    print(f"Processed max: {torch.max(torch.abs(processed)).item():.6f}")
    
    return processed


def compare_preprocessing():
    """Compare preprocessing methods."""
    # Generate test audio
    audio, sr = generate_test_audio(duration=3.0)
    
    # Test both methods
    orig_processed = test_original_preprocessing(audio, sr)
    onnx_processed = test_onnx_preprocessing(audio, sr)
    
    # Convert to tensors for comparison
    if isinstance(orig_processed, np.ndarray):
        orig_processed = torch.from_numpy(orig_processed)
    
    # Compare
    print("\n=== Comparison ===")
    print(f"Shape match: {orig_processed.shape == onnx_processed.shape}")
    
    # Compute differences
    diff = orig_processed - onnx_processed
    mse = torch.mean(diff ** 2)
    max_diff = torch.max(torch.abs(diff))
    
    # Compute relative error
    orig_power = torch.mean(orig_processed ** 2)
    relative_error = mse / (orig_power + 1e-8)
    
    print(f"MSE: {mse.item():.8f}")
    print(f"Max absolute difference: {max_diff.item():.8f}")
    print(f"Relative error: {relative_error.item():.8f}")
    print(f"SNR: {10 * torch.log10(orig_power / (mse + 1e-8)).item():.2f} dB")
    
    # Save outputs
    output_dir = Path("outputs/preprocessing_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and save
    def save_audio(tensor, path):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim == 2:
            tensor = tensor.T
        else:
            tensor = tensor.reshape(-1, 1)
        sf.write(path, tensor, sr)
    
    save_audio(audio, output_dir / "input.wav")
    save_audio(orig_processed, output_dir / "vampnet_preprocessed.wav")
    save_audio(onnx_processed, output_dir / "onnx_preprocessed.wav")
    
    print(f"\nAudio saved to {output_dir}")


if __name__ == "__main__":
    compare_preprocessing()