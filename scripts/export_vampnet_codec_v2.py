#!/usr/bin/env python3
"""
Export VampNet codec to ONNX with proper dynamic sequence handling.
This version ensures the encoder produces the correct number of tokens for any input length.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))

import vampnet
from audiotools import AudioSignal


class VampNetEncoderWrapper(torch.nn.Module):
    """Minimal wrapper that properly handles dynamic sequences."""
    
    def __init__(self, interface):
        super().__init__()
        self.codec = interface.codec
        self.device = interface.device
        self.sample_rate = self.codec.sample_rate
        self.hop_length = self.codec.hop_length
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to tokens. 
        Args:
            audio: [batch, channels, samples] at 44100Hz
        Returns:
            codes: [batch, n_codebooks, time_steps]
        """
        # Move to device
        audio = audio.to(self.device)
        
        # The codec expects the audio to be padded to multiple of hop_length
        batch, channels, length = audio.shape
        
        # Calculate padding
        remainder = length % self.hop_length
        if remainder > 0:
            pad_length = self.hop_length - remainder
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        # Encode using the codec
        with torch.no_grad():
            # Call encode with audio and sample_rate
            result = self.codec.encode(audio, self.sample_rate)
            
            # Extract codes
            if isinstance(result, dict):
                codes = result["codes"]
            else:
                codes = result
                
        return codes


def export_encoder_v2():
    """Export VampNet encoder with proper dynamic handling."""
    print("=== Exporting VampNet Encoder V2 ===\n")
    
    # Load VampNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    print(f"Codec info:")
    print(f"  Sample rate: {interface.codec.sample_rate}")
    print(f"  Hop length: {interface.codec.hop_length}")
    print(f"  Codebooks: {interface.codec.n_codebooks}")
    
    # Create wrapper
    encoder = VampNetEncoderWrapper(interface)
    encoder.eval()
    
    # Test with PyTorch first
    print("\nTesting with PyTorch:")
    for seconds in [1, 2, 5, 10]:
        samples = seconds * 44100
        test_audio = torch.randn(1, 1, samples).to(device)
        with torch.no_grad():
            codes = encoder(test_audio)
        expected = samples // interface.codec.hop_length
        print(f"  {seconds}s ({samples} samples): {codes.shape} (expected ~{expected} tokens)")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Use a medium-sized example for export
    dummy_input = torch.randn(1, 1, 5 * 44100).to(device)
    
    onnx_path = output_dir / "vampnet_codec_encoder_v2.onnx"
    
    # Export with dynamic axes
    torch.onnx.export(
        encoder,
        dummy_input,
        str(onnx_path),
        input_names=['audio'],
        output_names=['codes'],
        dynamic_axes={
            'audio': {0: 'batch', 2: 'samples'},  # batch and time are dynamic
            'codes': {0: 'batch', 2: 'time'}      # batch and time are dynamic
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ Exported to: {onnx_path}")
    
    # Test ONNX model
    print("\nTesting ONNX model:")
    session = ort.InferenceSession(str(onnx_path))
    
    for seconds in [1, 2, 5, 10]:
        samples = seconds * 44100
        test_audio = np.random.randn(1, 1, samples).astype(np.float32)
        codes = session.run(None, {'audio': test_audio})[0]
        expected = samples // interface.codec.hop_length
        print(f"  {seconds}s: {codes.shape} (expected ~{expected} tokens)")
        
        # Check if it matches expected
        if abs(codes.shape[2] - expected) <= 1:
            print(f"    ✓ Correct!")
        else:
            print(f"    ✗ Wrong! Got {codes.shape[2]}, expected {expected}")
    
    # Test with real audio
    audio_path = Path("../assets/example.wav")
    if audio_path.exists():
        print(f"\nTesting with real audio: {audio_path}")
        audio_np, sr = sf.read(str(audio_path))
        
        # Test with VampNet
        sig = AudioSignal(audio_np[np.newaxis, :], sample_rate=sr)
        z_vampnet = interface.encode(sig)
        print(f"  VampNet: {z_vampnet.shape}")
        
        # Test with ONNX
        audio_input = audio_np[np.newaxis, np.newaxis, :].astype(np.float32)
        codes_onnx = session.run(None, {'audio': audio_input})[0]
        print(f"  ONNX: {codes_onnx.shape}")
        
        if codes_onnx.shape[2] == z_vampnet.shape[2]:
            print(f"  ✓ Sequence lengths match!")
            
            # Compare actual tokens
            z_np = z_vampnet.cpu().numpy()
            match_rate = (codes_onnx == z_np).mean() * 100
            print(f"  Token match rate: {match_rate:.1f}%")
        else:
            print(f"  ✗ Sequence length mismatch!")
    
    print("\n✓ Export complete!")
    return onnx_path


if __name__ == "__main__":
    export_encoder_v2()