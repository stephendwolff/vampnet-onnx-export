"""
Test inference with exported ONNX models.
"""

import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path

# Import vampnet for preprocessing
import vampnet
import audiotools as at
from vampnet_onnx.audio_processor import AudioProcessor


def test_onnx_inference():
    """Test the exported ONNX models."""
    print("=== Testing ONNX Model Inference ===")
    
    # Paths to ONNX models
    encoder_path = "onnx_models/vampnet_codec/encoder.onnx"
    decoder_path = "onnx_models/vampnet_codec/decoder.onnx"
    
    if not Path(encoder_path).exists() or not Path(decoder_path).exists():
        print("❌ ONNX models not found. Run test_onnx_export_codec.py first.")
        return
    
    # Create ONNX Runtime sessions
    print("\nLoading ONNX models...")
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)
    
    # Print model info
    print("\nEncoder inputs:", [inp.name for inp in encoder_session.get_inputs()])
    print("Encoder outputs:", [out.name for out in encoder_session.get_outputs()])
    print("Decoder inputs:", [inp.name for inp in decoder_session.get_inputs()])
    print("Decoder outputs:", [out.name for out in decoder_session.get_outputs()])
    
    # Generate test audio
    print("\n--- Generating Test Audio ---")
    duration = 2.0
    sr = 44100
    t = torch.linspace(0, duration, int(sr * duration))
    freq1, freq2 = 440, 554  # A4, C#5
    audio = 0.3 * (torch.sin(2 * np.pi * freq1 * t) + 
                   torch.sin(2 * np.pi * freq2 * t) * 0.8)
    audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Preprocess audio
    print("\n--- Preprocessing Audio ---")
    # Load interface just for preprocessing
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth"
    )
    
    # Use AudioProcessor for consistent preprocessing
    processor = AudioProcessor(
        target_sample_rate=44100,
        target_loudness=-24.0,
        hop_length=interface.codec.hop_length
    )
    
    processed_audio = processor(audio)
    print(f"Processed audio shape: {processed_audio.shape}")
    
    # Convert to numpy for ONNX
    audio_numpy = processed_audio.numpy()
    
    # Run encoder
    print("\n--- Running ONNX Encoder ---")
    encoder_output = encoder_session.run(None, {'audio': audio_numpy})
    codes = encoder_output[0]
    print(f"Encoded codes shape: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    print(f"Codes range: [{codes.min()}, {codes.max()}]")
    
    # Run decoder
    print("\n--- Running ONNX Decoder ---")
    decoder_output = decoder_session.run(None, {'codes': codes})
    reconstructed = decoder_output[0]
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    print(f"Reconstructed dtype: {reconstructed.dtype}")
    
    # Compare with PyTorch version
    print("\n--- Comparing with PyTorch ---")
    from vampnet_onnx.vampnet_codec import VampNetCodecEncoder, VampNetCodecDecoder
    
    encoder_torch = VampNetCodecEncoder(interface.codec)
    decoder_torch = VampNetCodecDecoder(interface.codec)
    
    with torch.no_grad():
        codes_torch = encoder_torch(processed_audio).numpy()
        reconstructed_torch = decoder_torch(torch.from_numpy(codes)).numpy()
    
    # Check if codes match
    codes_match = np.allclose(codes, codes_torch)
    print(f"Codes match: {codes_match}")
    
    # Check reconstruction quality
    mse = np.mean((reconstructed - reconstructed_torch) ** 2)
    signal_power = np.mean(reconstructed_torch ** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-8))
    
    print(f"Reconstruction MSE: {mse:.8f}")
    print(f"Reconstruction SNR: {snr:.2f} dB")
    
    # Save outputs
    output_dir = Path("outputs/onnx_inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_audio(array, path):
        if array.ndim == 3:
            array = array[0]
        if array.ndim == 2:
            array = array.T
        else:
            array = array.reshape(-1, 1)
        sf.write(path, array, sr)
    
    save_audio(processed_audio.numpy(), output_dir / "input.wav")
    save_audio(reconstructed, output_dir / "onnx_reconstructed.wav")
    save_audio(reconstructed_torch, output_dir / "torch_reconstructed.wav")
    
    print(f"\nAudio saved to {output_dir}")
    
    if codes_match and snr > 30:
        print("\n✅ ONNX models working correctly!")
    else:
        print("\n⚠️ ONNX models have some differences from PyTorch")


if __name__ == "__main__":
    test_onnx_inference()