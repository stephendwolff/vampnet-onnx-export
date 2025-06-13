"""
Direct test of codec encode/decode functionality.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Import vampnet components
import vampnet
import audiotools as at

# Import our ONNX export components
from vampnet_onnx.vampnet_codec import VampNetCodecEncoder, VampNetCodecDecoder


def test_codec_direct():
    """Test codec directly with preprocessed audio."""
    print("=== Testing Codec Directly ===")
    
    # Load interface and codec
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth"
    )
    codec = interface.codec
    
    print(f"Codec info:")
    print(f"  Sample rate: {codec.sample_rate}")
    print(f"  Hop length: {codec.hop_length}")
    print(f"  Num codebooks: {codec.n_codebooks}")
    
    # Generate test audio
    duration = 2.0
    sr = 44100
    t = torch.linspace(0, duration, int(sr * duration))
    freq1, freq2 = 440, 554  # A4, C#5
    audio = 0.3 * (torch.sin(2 * np.pi * freq1 * t) + 
                   torch.sin(2 * np.pi * freq2 * t) * 0.8)
    audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Preprocess using original vampnet method
    signal = at.AudioSignal(audio.numpy(), sr)
    signal = interface._preprocess(signal)
    preprocessed = signal.samples
    print(f"\nPreprocessed shape: {preprocessed.shape}")
    
    # Test 1: Original codec encode/decode
    print("\n--- Original Codec ---")
    with torch.no_grad():
        # Encode
        encoded = codec.encode(preprocessed, signal.sample_rate)
        codes = encoded["codes"]
        print(f"Encoded codes shape: {codes.shape}")
        print(f"Codes dtype: {codes.dtype}")
        print(f"Codes range: [{codes.min().item()}, {codes.max().item()}]")
        
        # Decode: first convert codes to embeddings, then decode
        z_q, _, _ = codec.quantizer.from_codes(codes)
        reconstructed = codec.decode(z_q)
        if isinstance(reconstructed, dict):
            reconstructed = reconstructed["audio"]
        
        print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test 2: ONNX wrapper encode/decode
    print("\n--- ONNX Wrapper ---")
    encoder = VampNetCodecEncoder(codec)
    decoder = VampNetCodecDecoder(codec)
    
    with torch.no_grad():
        # Encode
        onnx_codes = encoder(preprocessed)
        print(f"ONNX codes shape: {onnx_codes.shape}")
        print(f"ONNX codes dtype: {onnx_codes.dtype}")
        print(f"ONNX codes range: [{onnx_codes.min().item()}, {onnx_codes.max().item()}]")
        
        # Check if codes match
        codes_match = torch.allclose(codes, onnx_codes)
        print(f"Codes match: {codes_match}")
        
        # Decode
        onnx_reconstructed = decoder(onnx_codes)
        print(f"ONNX reconstructed shape: {onnx_reconstructed.shape}")
    
    # Compare outputs
    print("\n--- Comparison ---")
    # Ensure same shape for comparison
    min_len = min(reconstructed.shape[-1], onnx_reconstructed.shape[-1])
    orig_recon = reconstructed[..., :min_len]
    onnx_recon = onnx_reconstructed[..., :min_len]
    
    mse = torch.mean((orig_recon - onnx_recon) ** 2)
    signal_power = torch.mean(orig_recon ** 2)
    snr = 10 * torch.log10(signal_power / (mse + 1e-8))
    
    print(f"MSE: {mse.item():.8f}")
    print(f"SNR: {snr.item():.2f} dB")
    print(f"Reconstructions match: {torch.allclose(orig_recon, onnx_recon, rtol=1e-3)}")
    
    # Save outputs
    output_dir = Path("outputs/codec_direct_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    save_audio(preprocessed, output_dir / "preprocessed.wav")
    save_audio(reconstructed, output_dir / "original_reconstructed.wav")
    save_audio(onnx_reconstructed, output_dir / "onnx_reconstructed.wav")
    
    print(f"\nAudio saved to {output_dir}")
    
    return codes_match, torch.allclose(orig_recon, onnx_recon, rtol=1e-3)


if __name__ == "__main__":
    codes_match, recon_match = test_codec_direct()
    
    if codes_match and recon_match:
        print("\n✅ ONNX codec wrapper working correctly!")
    else:
        print("\n❌ ONNX codec wrapper has differences")