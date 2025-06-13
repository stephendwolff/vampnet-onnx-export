#!/usr/bin/env python3
"""
Export VampNet codec using TorchScript as an intermediate step.
This might preserve dynamic behavior better than direct ONNX export.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import vampnet
from audiotools import AudioSignal


class CodecEncoderWrapper(torch.nn.Module):
    """Wrapper that handles padding dynamically."""
    
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.hop_length = codec.hop_length
        self.sample_rate = codec.sample_rate
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to codes with dynamic padding.
        Args:
            audio: [batch, channels, samples]
        Returns:
            codes: [batch, n_codebooks, time_steps]
        """
        batch, channels, length = audio.shape
        
        # Calculate required padding dynamically
        # Use operations that ONNX can handle
        remainder = length % self.hop_length
        pad_length = (self.hop_length - remainder) % self.hop_length
        
        # Apply padding if needed
        if pad_length > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_length), mode='constant', value=0)
        
        # Encode
        result = self.codec.encode(audio, self.sample_rate)
        codes = result["codes"] if isinstance(result, dict) else result
        
        return codes


def export_with_torchscript():
    """Export using TorchScript as intermediate."""
    print("=== Exporting VampNet Codec via TorchScript ===\n")
    
    # Load VampNet
    device = 'cpu'  # Use CPU for export
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    print(f"Codec info:")
    print(f"  Type: {type(interface.codec)}")
    print(f"  Sample rate: {interface.codec.sample_rate}")
    print(f"  Hop length: {interface.codec.hop_length}")
    
    # Create wrapper
    encoder = CodecEncoderWrapper(interface.codec)
    encoder.eval()
    
    # Test with different sizes first
    print("\nTesting PyTorch model:")
    test_sizes = [44100, 88200, 132300, 220500, 441000]
    
    for size in test_sizes:
        audio = torch.randn(1, 1, size)
        with torch.no_grad():
            codes = encoder(audio)
        expected = size // interface.codec.hop_length
        print(f"  {size:6d} samples: {codes.shape[2]:3d} tokens (expected {expected:3d})")
    
    # Try TorchScript export
    print("\nAttempting TorchScript export...")
    
    try:
        # Method 1: Try scripting
        print("  Trying torch.jit.script...")
        scripted = torch.jit.script(encoder)
        print("    ✓ Script successful!")
        
        # Test scripted model
        test_audio = torch.randn(1, 1, 88200)
        scripted_output = scripted(test_audio)
        print(f"    Scripted output: {scripted_output.shape}")
        
    except Exception as e:
        print(f"    ✗ Script failed: {e}")
        
        # Method 2: Try tracing with multiple examples
        print("\n  Trying torch.jit.trace with multiple examples...")
        try:
            # Create multiple example inputs
            example_inputs = [
                torch.randn(1, 1, 44100),
                torch.randn(1, 1, 88200),
                torch.randn(1, 1, 132300),
            ]
            
            # Trace with first example
            traced = torch.jit.trace(encoder, example_inputs[0], check_inputs=example_inputs)
            print("    ✓ Trace successful!")
            
            # Test traced model
            for inp in example_inputs:
                out = traced(inp)
                print(f"    Input {inp.shape} -> Output {out.shape}")
                
            scripted = traced
            
        except Exception as e:
            print(f"    ✗ Trace failed: {e}")
            return
    
    # Export to ONNX from TorchScript
    print("\nExporting TorchScript to ONNX...")
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Use medium example
    dummy_input = torch.randn(1, 1, 5 * 44100)
    
    onnx_path = output_dir / "vampnet_codec_encoder_torchscript.onnx"
    
    try:
        torch.onnx.export(
            scripted,
            dummy_input,
            str(onnx_path),
            input_names=['audio'],
            output_names=['codes'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'samples'},
                'codes': {0: 'batch', 2: 'time'}
            },
            opset_version=17,
            export_params=True
        )
        print(f"✓ Exported to: {onnx_path}")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return
    
    # Test ONNX model
    print("\nTesting ONNX model:")
    session = ort.InferenceSession(str(onnx_path))
    
    for size in test_sizes:
        audio_np = np.random.randn(1, 1, size).astype(np.float32)
        codes = session.run(None, {'audio': audio_np})[0]
        expected = size // interface.codec.hop_length
        is_correct = abs(codes.shape[2] - expected) <= 1
        status = "✓" if is_correct else "✗"
        print(f"  {size:6d} samples: {codes.shape[2]:3d} tokens (expected {expected:3d}) {status}")
    
    return onnx_path


def export_minimal_encoder():
    """Try exporting just the core encoder without preprocessing."""
    print("\n\n=== Exporting Minimal Encoder ===\n")
    
    # Load VampNet
    device = 'cpu'
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    # We'll handle padding externally and just export the core encoding
    class MinimalEncoder(torch.nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.encoder = codec.encoder
            self.quantizer = codec.quantizer
            
        def forward(self, audio_padded: torch.Tensor) -> torch.Tensor:
            """
            Encode pre-padded audio.
            Args:
                audio_padded: [batch, 1, samples] - must be padded to multiple of 768
            Returns:
                codes: [batch, n_codebooks, time_steps]
            """
            # Encode to latent
            latent = self.encoder(audio_padded)
            
            # Quantize
            quantized, codes, _, _, _ = self.quantizer(latent)
            
            return codes
    
    encoder = MinimalEncoder(interface.codec)
    encoder.eval()
    
    # Test it
    print("Testing minimal encoder:")
    # Pre-pad audio to multiples of 768
    for seconds in [1, 2, 5]:
        samples = seconds * 44100
        padded_samples = ((samples + 767) // 768) * 768
        audio = torch.randn(1, 1, padded_samples)
        
        with torch.no_grad():
            codes = encoder(audio)
        
        print(f"  {seconds}s ({padded_samples} padded samples): {codes.shape}")
    
    # Export this minimal version
    dummy_padded = torch.randn(1, 1, 768 * 288)  # Pre-padded
    
    onnx_path = Path("models/vampnet_codec_encoder_minimal.onnx")
    
    torch.onnx.export(
        encoder,
        dummy_padded,
        str(onnx_path),
        input_names=['audio_padded'],
        output_names=['codes'],
        dynamic_axes={
            'audio_padded': {0: 'batch', 2: 'samples'},
            'codes': {0: 'batch', 2: 'time'}
        },
        opset_version=17
    )
    
    print(f"\n✓ Exported minimal encoder to: {onnx_path}")
    print("Note: This encoder requires pre-padded input (multiple of 768 samples)")
    
    # Test ONNX
    print("\nTesting minimal ONNX encoder:")
    session = ort.InferenceSession(str(onnx_path))
    
    for seconds in [1, 2, 5, 10]:
        samples = seconds * 44100
        padded_samples = ((samples + 767) // 768) * 768
        audio_np = np.random.randn(1, 1, padded_samples).astype(np.float32)
        
        codes = session.run(None, {'audio_padded': audio_np})[0]
        expected = padded_samples // 768
        
        print(f"  {seconds}s ({padded_samples} samples): {codes.shape[2]} tokens (expected {expected})")


if __name__ == "__main__":
    # Try TorchScript approach
    torchscript_path = export_with_torchscript()
    
    # Try minimal encoder approach  
    export_minimal_encoder()