#!/usr/bin/env python3
"""
Test script for codec export functionality.
Tests both simplified and full VampNet codec if available.
"""

import numpy as np
import torch
import os
from vampnet_onnx.exporters import export_codec_encoder, export_codec_decoder, VAMPNET_AVAILABLE

def test_simplified_codec():
    """Test the simplified codec export and inference."""
    print("Testing simplified codec export...")
    
    # Export encoder
    encoder_path = "models/test_codec_encoder_simplified.onnx"
    export_codec_encoder(
        encoder_path,
        use_simplified=True,
        example_audio_length=44100  # 1 second
    )
    
    # Export decoder
    decoder_path = "models/test_codec_decoder_simplified.onnx"
    export_codec_decoder(
        decoder_path,
        use_simplified=True,
        example_sequence_length=57  # ~1 second at hop_length=768
    )
    
    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort
        
        # Test encoder
        print("\nTesting encoder inference...")
        encoder_session = ort.InferenceSession(encoder_path)
        test_audio = np.random.randn(1, 1, 44100).astype(np.float32)
        
        encoder_output = encoder_session.run(None, {'audio': test_audio})
        codes = encoder_output[0]
        print(f"Encoder output shape: {codes.shape}")
        print(f"Codes dtype: {codes.dtype}")
        
        # Test decoder
        print("\nTesting decoder inference...")
        decoder_session = ort.InferenceSession(decoder_path)
        
        decoder_output = decoder_session.run(None, {'codes': codes})
        reconstructed_audio = decoder_output[0]
        print(f"Decoder output shape: {reconstructed_audio.shape}")
        print(f"Audio range: [{reconstructed_audio.min():.3f}, {reconstructed_audio.max():.3f}]")
        
        print("\nSimplified codec test passed!")
        
    except Exception as e:
        print(f"Error during inference: {e}")


def test_vampnet_codec():
    """Test the full VampNet codec export if available."""
    if not VAMPNET_AVAILABLE:
        print("\nVampNet not available, skipping full codec test")
        return
        
    print("\nTesting VampNet codec export...")
    
    try:
        # Export encoder
        encoder_path = "models/test_codec_encoder_vampnet.onnx"
        export_codec_encoder(
            encoder_path,
            use_simplified=False,
            use_vampnet=True,
            example_audio_length=44100  # 1 second
        )
        
        # Export decoder
        decoder_path = "models/test_codec_decoder_vampnet.onnx"
        export_codec_decoder(
            decoder_path,
            use_simplified=False,
            use_vampnet=True,
            example_sequence_length=57  # ~1 second at hop_length=768
        )
        
        print("VampNet codec export successful!")
        
        # Test inference
        import onnxruntime as ort
        
        # Test encoder
        print("\nTesting VampNet encoder inference...")
        encoder_session = ort.InferenceSession(encoder_path)
        test_audio = np.random.randn(1, 1, 44100).astype(np.float32)
        
        encoder_output = encoder_session.run(None, {'audio': test_audio})
        codes = encoder_output[0]
        print(f"VampNet encoder output shape: {codes.shape}")
        
        # Test decoder
        print("\nTesting VampNet decoder inference...")
        decoder_session = ort.InferenceSession(decoder_path)
        
        decoder_output = decoder_session.run(None, {'codes': codes})
        reconstructed_audio = decoder_output[0]
        print(f"VampNet decoder output shape: {reconstructed_audio.shape}")
        
        print("\nVampNet codec test passed!")
        
    except Exception as e:
        print(f"Error with VampNet codec: {e}")


def compare_audio_quality():
    """Compare audio quality between simplified and full codec."""
    print("\n" + "="*50)
    print("Audio Quality Comparison")
    print("="*50)
    
    # Generate test audio
    duration = 3.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a test signal with multiple frequencies
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
        0.2 * np.sin(2 * np.pi * 659.25 * t) +  # E5
        0.1 * np.random.randn(len(t)) * 0.1  # Light noise
    )
    
    # Reshape for model input
    test_audio = test_audio.astype(np.float32).reshape(1, 1, -1)
    
    print(f"Test audio shape: {test_audio.shape}")
    print(f"Test audio range: [{test_audio.min():.3f}, {test_audio.max():.3f}]")
    
    # Save original audio
    import soundfile as sf
    os.makedirs("outputs", exist_ok=True)
    sf.write("outputs/test_original.wav", test_audio[0, 0], sample_rate)
    print("\nSaved original audio to outputs/test_original.wav")
    
    # Test with simplified codec
    try:
        import onnxruntime as ort
        
        if os.path.exists("models/test_codec_encoder_simplified.onnx"):
            print("\nProcessing with simplified codec...")
            encoder = ort.InferenceSession("models/test_codec_encoder_simplified.onnx")
            decoder = ort.InferenceSession("models/test_codec_decoder_simplified.onnx")
            
            # Encode
            codes = encoder.run(None, {'audio': test_audio})[0]
            print(f"Codes shape: {codes.shape}")
            
            # Decode
            reconstructed = decoder.run(None, {'codes': codes})[0]
            
            # Save reconstructed audio
            sf.write("outputs/test_simplified_codec.wav", reconstructed[0, 0], sample_rate)
            print("Saved simplified codec output to outputs/test_simplified_codec.wav")
            
            # Calculate SNR
            signal_power = np.mean(test_audio ** 2)
            noise_power = np.mean((test_audio - reconstructed) ** 2)
            snr = 10 * np.log10(signal_power / noise_power)
            print(f"Simplified codec SNR: {snr:.2f} dB")
            
    except Exception as e:
        print(f"Error with simplified codec: {e}")
    
    # Test with VampNet codec if available
    if VAMPNET_AVAILABLE and os.path.exists("models/test_codec_encoder_vampnet.onnx"):
        try:
            print("\nProcessing with VampNet codec...")
            encoder = ort.InferenceSession("models/test_codec_encoder_vampnet.onnx")
            decoder = ort.InferenceSession("models/test_codec_decoder_vampnet.onnx")
            
            # Encode
            codes = encoder.run(None, {'audio': test_audio})[0]
            print(f"VampNet codes shape: {codes.shape}")
            
            # Decode
            reconstructed = decoder.run(None, {'codes': codes})[0]
            
            # Save reconstructed audio
            sf.write("outputs/test_vampnet_codec.wav", reconstructed[0, 0], sample_rate)
            print("Saved VampNet codec output to outputs/test_vampnet_codec.wav")
            
            # Calculate SNR
            signal_power = np.mean(test_audio ** 2)
            noise_power = np.mean((test_audio - reconstructed) ** 2)
            snr = 10 * np.log10(signal_power / noise_power)
            print(f"VampNet codec SNR: {snr:.2f} dB")
            
        except Exception as e:
            print(f"Error with VampNet codec: {e}")


if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run tests
    test_simplified_codec()
    test_vampnet_codec()
    compare_audio_quality()
    
    print("\n" + "="*50)
    print("Testing complete!")
    print("Check the outputs/ directory for generated audio files.")
    print("="*50)