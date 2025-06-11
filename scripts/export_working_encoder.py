#!/usr/bin/env python3
"""Export a working ONNX encoder using the correct codec.encode method."""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import vampnet


class VampNetEncoderONNX(torch.nn.Module):
    """ONNX-compatible encoder using codec.encode."""
    
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.sample_rate = codec.sample_rate
        self.hop_length = codec.hop_length
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to codes.
        Args:
            audio: [batch, 1, samples] at 44100Hz
        Returns:
            codes: [batch, n_codebooks, time_steps]
        """
        # The codec.encode handles padding internally
        result = self.codec.encode(audio, self.sample_rate)
        
        # Extract codes from the result dict
        codes = result["codes"]
        
        return codes


def test_and_export():
    """Test and export the encoder."""
    print("=== Exporting Working VampNet Encoder ===\n")
    
    # Load VampNet
    device = 'cpu'
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
    
    # Create encoder wrapper
    encoder = VampNetEncoderONNX(interface.codec)
    encoder.eval()
    
    # Test with PyTorch first
    print("\nTesting with PyTorch:")
    test_correct = True
    
    for seconds in [0.5, 1, 2, 3, 5, 10]:
        samples = int(seconds * 44100)
        test_audio = torch.randn(1, 1, samples)
        
        with torch.no_grad():
            codes = encoder(test_audio)
        
        expected = samples // interface.codec.hop_length
        is_correct = abs(codes.shape[2] - expected) <= 1
        status = "✓" if is_correct else "✗"
        test_correct &= is_correct
        
        print(f"  {seconds:4.1f}s ({samples:6d} samples): {codes.shape[2]:3d} tokens (expected ~{expected:3d}) {status}")
    
    if not test_correct:
        print("\n❌ PyTorch model doesn't work correctly!")
        return None
    
    print("\n✅ PyTorch model works correctly!")
    
    # Try different export approaches
    print("\n--- Attempting ONNX Export ---")
    
    # Method 1: Direct export with dynamic axes
    print("\nMethod 1: Direct ONNX export with dynamic axes")
    
    output_dir = Path("models") 
    output_dir.mkdir(exist_ok=True)
    
    # Use a variety of input sizes for better tracing
    dummy_inputs = [
        torch.randn(1, 1, int(s * 44100)) 
        for s in [1, 2, 3, 5, 7]
    ]
    
    onnx_path = output_dir / "vampnet_encoder_working.onnx"
    
    try:
        # Try with check_trace_inputs for better dynamic behavior
        torch.onnx.export(
            encoder,
            dummy_inputs[2],  # Use 3s as base
            str(onnx_path),
            input_names=['audio'],
            output_names=['codes'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'samples'},
                'codes': {0: 'batch', 2: 'time'}
            },
            opset_version=17,
            do_constant_folding=False,  # Try without constant folding
            export_params=True,
            verbose=False
        )
        print(f"✓ Exported to: {onnx_path}")
        
        # Test the ONNX model
        print("\nTesting ONNX model:")
        session = ort.InferenceSession(str(onnx_path))
        
        all_correct = True
        for seconds in [0.5, 1, 2, 3, 5, 10]:
            samples = int(seconds * 44100)
            test_audio_np = np.random.randn(1, 1, samples).astype(np.float32)
            
            codes_onnx = session.run(None, {'audio': test_audio_np})[0]
            expected = samples // interface.codec.hop_length
            is_correct = abs(codes_onnx.shape[2] - expected) <= 1
            status = "✓" if is_correct else "✗"
            all_correct &= is_correct
            
            print(f"  {seconds:4.1f}s: {codes_onnx.shape[2]:3d} tokens (expected ~{expected:3d}) {status}")
        
        if all_correct:
            print("\n🎉 SUCCESS! The ONNX encoder works correctly!")
            return onnx_path
        else:
            print("\n❌ ONNX model has fixed output size")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Method 2: Export without the internal padding
    print("\n\nMethod 2: Pre-padded encoder")
    
    class PrePaddedEncoder(torch.nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.codec = codec
            
        def forward(self, audio_padded: torch.Tensor) -> torch.Tensor:
            """Encode pre-padded audio (must be multiple of 768)."""
            # Skip internal padding by calling encode directly
            result = self.codec.encode(audio_padded, 44100)
            return result["codes"]
    
    encoder2 = PrePaddedEncoder(interface.codec)
    encoder2.eval()
    
    # Test pre-padded version
    print("\nTesting pre-padded encoder:")
    test_correct = True
    
    for seconds in [1, 2, 5]:
        samples = int(seconds * 44100)
        padded_samples = ((samples + 767) // 768) * 768
        test_audio = torch.randn(1, 1, padded_samples)
        
        with torch.no_grad():
            codes = encoder2(test_audio)
        
        expected = padded_samples // 768
        is_correct = codes.shape[2] == expected
        status = "✓" if is_correct else "✗"
        test_correct &= is_correct
        
        print(f"  {seconds}s ({padded_samples} padded): {codes.shape[2]} tokens (expected {expected}) {status}")
    
    if test_correct:
        # Export pre-padded version
        onnx_path2 = output_dir / "vampnet_encoder_prepadded.onnx"
        
        dummy_padded = torch.randn(1, 1, 768 * 100)
        
        torch.onnx.export(
            encoder2,
            dummy_padded,
            str(onnx_path2),
            input_names=['audio_padded'],
            output_names=['codes'],
            dynamic_axes={
                'audio_padded': {0: 'batch', 2: 'samples_padded'},
                'codes': {0: 'batch', 2: 'time'}
            },
            opset_version=17,
            export_params=True
        )
        
        print(f"\n✓ Exported pre-padded version to: {onnx_path2}")
        print("\nNote: This version requires input padded to multiples of 768")
        
        return onnx_path2
    
    return None


if __name__ == "__main__":
    result = test_and_export()
    if result:
        print(f"\n✅ Successfully exported working encoder to: {result}")
    else:
        print("\n❌ Failed to export a working encoder")