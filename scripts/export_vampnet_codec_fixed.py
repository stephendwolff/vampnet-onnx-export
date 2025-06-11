#!/usr/bin/env python3
"""
Export the VampNet codec models to ONNX format - Fixed version.
This ensures the ONNX encoder produces identical tokens to VampNet.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import soundfile as sf
from audiotools import AudioSignal

sys.path.append(str(Path(__file__).parent.parent))

import vampnet

class VampNetEncoderONNX(torch.nn.Module):
    """Fixed ONNX-compatible VampNet encoder wrapper."""
    
    def __init__(self, codec_model, device='cpu'):
        super().__init__()
        self.codec = codec_model
        self.device = device
        self.sample_rate = codec_model.sample_rate
        self.hop_length = codec_model.hop_length
        self.n_codebooks = codec_model.n_codebooks
        
        # Move codec to device
        self.codec.to(device)
        self.codec.eval()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete tokens.
        
        Args:
            audio: Audio tensor [batch, 1, samples] at 44100Hz
            
        Returns:
            Token codes [batch, n_codebooks, sequence_length]
        """
        with torch.no_grad():
            # Apply codec preprocessing (padding to multiple of hop_length)
            preprocessed, original_length = self.codec.preprocess(audio, self.sample_rate)
            
            # Encode
            result = self.codec.encode(preprocessed, self.sample_rate)
            
            # Extract codes
            if isinstance(result, dict):
                codes = result["codes"]
            else:
                codes = result
                
            return codes


class VampNetDecoderONNX(torch.nn.Module):
    """ONNX-compatible VampNet decoder wrapper."""
    
    def __init__(self, codec_model, device='cpu'):
        super().__init__()
        self.codec = codec_model
        self.device = device
        self.sample_rate = codec_model.sample_rate
        self.hop_length = codec_model.hop_length
        
        # Move codec to device
        self.codec.to(device)
        self.codec.eval()
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to audio.
        
        Args:
            codes: Token codes [batch, n_codebooks, sequence_length]
            
        Returns:
            Decoded audio [batch, 1, samples]
        """
        with torch.no_grad():
            # Convert codes to embeddings using the quantizer
            z_q, _, _ = self.codec.quantizer.from_codes(codes)
            
            # Decode embeddings to audio
            output = self.codec.decode(z_q)
            
            # Extract audio from output
            if isinstance(output, dict):
                audio = output["audio"]
            else:
                audio = output
            
            # Ensure output shape is [batch, 1, samples]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
                
            return audio


def export_vampnet_codec():
    """Export the VampNet codec to ONNX."""
    print("=== Exporting VampNet Codec to ONNX (Fixed) ===\n")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Load VampNet interface to get the actual codec
    print("Loading VampNet codec from pretrained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load with specific paths
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    codec_model = interface.codec
    print(f"✓ Loaded codec: {codec_model}")
    print(f"  Sample rate: {codec_model.sample_rate}")
    print(f"  Hop length: {codec_model.hop_length}")
    print(f"  Codebooks: {codec_model.n_codebooks}")
    
    # Test with real audio to verify
    print("\nTesting with real audio...")
    audio_path = Path("../assets/example.wav")
    if audio_path.exists():
        audio_np, sr = sf.read(audio_path)
        sig = AudioSignal(audio_np[np.newaxis, :], sample_rate=sr)
        
        # Encode with VampNet
        z_vampnet = interface.encode(sig)
        print(f"VampNet encoding shape: {z_vampnet.shape}")
    
    # Export encoder
    print("\n1. Exporting Encoder...")
    encoder_wrapper = VampNetEncoderONNX(codec_model, device=device)
    encoder_wrapper.eval()
    
    # Create example input - 3 seconds of audio
    example_audio = torch.randn(1, 1, 3 * 44100, device=device)
    
    encoder_path = output_dir / "vampnet_codec_encoder_fixed.onnx"
    
    torch.onnx.export(
        encoder_wrapper,
        example_audio,
        str(encoder_path),
        input_names=['audio'],
        output_names=['codes'],
        dynamic_axes={
            'audio': {2: 'samples'},
            'codes': {2: 'time'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ Encoder exported to: {encoder_path}")
    
    # Test encoder
    print("\n  Testing encoder...")
    encoder_session = ort.InferenceSession(str(encoder_path))
    
    # Test with various lengths
    for test_seconds in [1, 2, 3]:
        test_samples = test_seconds * 44100
        test_audio = np.random.randn(1, 1, test_samples).astype(np.float32)
        codes = encoder_session.run(None, {'audio': test_audio})[0]
        expected_tokens = test_samples // codec_model.hop_length
        print(f"    {test_seconds}s audio: {codes.shape[2]} tokens (expected ~{expected_tokens})")
    
    # Test with real audio if available
    if audio_path.exists() and 'audio_np' in locals():
        test_input = audio_np[np.newaxis, np.newaxis, :].astype(np.float32)
        codes_onnx = encoder_session.run(None, {'audio': test_input})[0]
        print(f"\n  Real audio test:")
        print(f"    VampNet: {z_vampnet.shape}")
        print(f"    ONNX: {codes_onnx.shape}")
        
        # Compare tokens
        if codes_onnx.shape[2] == z_vampnet.shape[2]:
            print(f"    ✓ Sequence lengths match!")
        else:
            print(f"    ✗ Sequence length mismatch!")
    
    # Export decoder
    print("\n2. Exporting Decoder...")
    decoder_wrapper = VampNetDecoderONNX(codec_model, device=device)
    decoder_wrapper.eval()
    
    # Use codes from encoder as example
    example_codes = torch.randint(0, 1024, (1, 14, 100), device=device)
    
    decoder_path = output_dir / "vampnet_codec_decoder_fixed.onnx"
    
    torch.onnx.export(
        decoder_wrapper,
        example_codes,
        str(decoder_path),
        input_names=['codes'],
        output_names=['audio'],
        dynamic_axes={
            'codes': {2: 'time'},
            'audio': {2: 'samples'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ Decoder exported to: {decoder_path}")
    
    # Test decoder
    print("\n  Testing decoder...")
    decoder_session = ort.InferenceSession(str(decoder_path))
    test_codes = np.random.randint(0, 1024, (1, 14, 58), dtype=np.int64)
    reconstructed = decoder_session.run(None, {'codes': test_codes})[0]
    print(f"    Decoded shape: {reconstructed.shape}")
    print(f"    Expected ~{58 * codec_model.hop_length} samples, got {reconstructed.shape[2]}")
    
    # Test round-trip
    print("\n3. Testing round-trip...")
    test_audio = np.random.randn(1, 1, 44100).astype(np.float32)
    
    # Encode
    codes_rt = encoder_session.run(None, {'audio': test_audio})[0]
    print(f"  Encoded: {codes_rt.shape}")
    
    # Decode
    audio_rt = decoder_session.run(None, {'codes': codes_rt})[0]
    print(f"  Decoded: {audio_rt.shape}")
    
    # Calculate MSE
    min_len = min(test_audio.shape[-1], audio_rt.shape[-1])
    mse = np.mean((test_audio[..., :min_len] - audio_rt[..., :min_len]) ** 2)
    print(f"  Round-trip MSE: {mse:.6f}")
    
    print("\n✓ VampNet codec export complete!")
    print(f"\nExported models:")
    print(f"  - {encoder_path}")
    print(f"  - {decoder_path}")
    
    return encoder_path, decoder_path


if __name__ == "__main__":
    export_vampnet_codec()