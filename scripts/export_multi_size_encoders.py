#!/usr/bin/env python3
"""Export multiple ONNX encoders for different input sizes."""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import vampnet


class FixedSizeEncoder(torch.nn.Module):
    """Encoder for a specific input size."""
    
    def __init__(self, codec, expected_samples):
        super().__init__()
        self.codec = codec
        self.expected_samples = expected_samples
        self.hop_length = codec.hop_length
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio of expected size."""
        # Verify input size
        assert audio.shape[2] == self.expected_samples, f"Expected {self.expected_samples} samples, got {audio.shape[2]}"
        
        # Encode
        result = self.codec.encode(audio, 44100)
        return result["codes"]


def export_multi_size():
    """Export encoders for multiple common sizes."""
    print("=== Exporting Multi-Size VampNet Encoders ===\n")
    
    # Load VampNet
    device = 'cpu'
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    output_dir = Path("models/vampnet_encoders")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Common audio lengths
    durations = {
        "1s": 44100,
        "2s": 88200,
        "3s": 132300,
        "5s": 220500,
        "10s": 441000,
        "15s": 661500,
        "30s": 1323000,
    }
    
    exported_models = {}
    
    for name, samples in durations.items():
        print(f"\nExporting encoder for {name} ({samples} samples)...")
        
        # Pad to multiple of hop_length
        padded_samples = ((samples + 767) // 768) * 768
        expected_tokens = padded_samples // 768
        
        # Create fixed-size encoder
        encoder = FixedSizeEncoder(interface.codec, padded_samples)
        encoder.eval()
        
        # Test
        test_input = torch.randn(1, 1, padded_samples)
        with torch.no_grad():
            codes = encoder(test_input)
        
        print(f"  Input: {padded_samples} samples")
        print(f"  Output: {codes.shape[2]} tokens (expected {expected_tokens})")
        
        if codes.shape[2] != expected_tokens:
            print(f"  ⚠️ Warning: Output size doesn't match expected!")
        
        # Export
        onnx_path = output_dir / f"encoder_{name}.onnx"
        
        torch.onnx.export(
            encoder,
            test_input,
            str(onnx_path),
            input_names=['audio'],
            output_names=['codes'],
            opset_version=17,
            export_params=True,
            verbose=False
        )
        
        # Test ONNX
        session = ort.InferenceSession(str(onnx_path))
        test_np = np.random.randn(1, 1, padded_samples).astype(np.float32)
        codes_onnx = session.run(None, {'audio': test_np})[0]
        
        if codes_onnx.shape[2] == expected_tokens:
            print(f"  ✓ ONNX test passed: {codes_onnx.shape}")
            exported_models[name] = {
                'path': str(onnx_path),
                'input_samples': padded_samples,
                'output_tokens': codes_onnx.shape[2]
            }
        else:
            print(f"  ✗ ONNX test failed: got {codes_onnx.shape[2]} tokens")
    
    # Save metadata
    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(exported_models, f, indent=2)
    
    print(f"\n\nExported {len(exported_models)} models to {output_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    # Create a helper class
    helper_code = '''
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json

class MultiSizeEncoder:
    """Helper to use the correct encoder based on input size."""
    
    def __init__(self, encoders_dir="models/vampnet_encoders"):
        self.encoders_dir = Path(encoders_dir)
        
        # Load metadata
        with open(self.encoders_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load all encoders
        self.sessions = {}
        for name, info in self.metadata.items():
            self.sessions[info['input_samples']] = ort.InferenceSession(info['path'])
    
    def encode(self, audio, sample_rate=44100):
        """Encode audio using the appropriate encoder."""
        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio[np.newaxis, np.newaxis, :]
        elif audio.ndim == 2:
            audio = audio[np.newaxis, :]
        
        # Pad to multiple of 768
        samples = audio.shape[2]
        padded_samples = ((samples + 767) // 768) * 768
        
        if padded_samples > samples:
            audio = np.pad(audio, ((0,0), (0,0), (0, padded_samples - samples)))
        
        # Find the right encoder
        if padded_samples in self.sessions:
            session = self.sessions[padded_samples]
            codes = session.run(None, {'audio': audio.astype(np.float32)})[0]
            return codes
        else:
            raise ValueError(f"No encoder for {padded_samples} samples. Available: {list(self.sessions.keys())}")
'''
    
    helper_path = output_dir / "multi_size_encoder.py"
    with open(helper_path, 'w') as f:
        f.write(helper_code)
    
    print(f"\nHelper class saved to {helper_path}")
    
    return exported_models


if __name__ == "__main__":
    models = export_multi_size()