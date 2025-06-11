#!/usr/bin/env python3
"""Export a minimal encoder that requires pre-padded input."""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import vampnet


class MinimalEncoder(torch.nn.Module):
    """Minimal encoder that expects pre-padded input."""
    
    def __init__(self, codec):
        super().__init__()
        self.encoder = codec.encoder
        self.quantizer = codec.quantizer
        self.hop_length = codec.hop_length
        
    def forward(self, audio_padded: torch.Tensor) -> torch.Tensor:
        """
        Encode pre-padded audio.
        Args:
            audio_padded: [batch, 1, samples] - MUST be multiple of hop_length (768)
        Returns:
            codes: [batch, n_codebooks, time_steps]
        """
        # Encode to latent representation
        latent = self.encoder(audio_padded)
        
        # Quantize to discrete codes
        # The quantizer returns: (quantized, codes, latents, commitment_loss, codebook_loss)
        _, codes, _, _, _ = self.quantizer(latent)
        
        return codes


def export_minimal():
    """Export minimal encoder that requires pre-padded input."""
    print("=== Exporting Minimal VampNet Encoder ===\n")
    
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
    
    # Create minimal encoder
    encoder = MinimalEncoder(interface.codec)
    encoder.eval()
    
    # Test with PyTorch
    print("\nTesting with PyTorch (pre-padded inputs):")
    for seconds in [1, 2, 5, 10]:
        samples = seconds * 44100
        # Round up to nearest multiple of hop_length
        padded_samples = ((samples + 767) // 768) * 768
        
        audio = torch.randn(1, 1, padded_samples)
        with torch.no_grad():
            codes = encoder(audio)
        
        expected_tokens = padded_samples // 768
        print(f"  {seconds}s ({padded_samples} padded samples): {codes.shape[2]} tokens (expected {expected_tokens})")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Use example with exact multiple of hop_length
    dummy_input = torch.randn(1, 1, 768 * 100)  # Exactly 100 tokens
    
    onnx_path = output_dir / "vampnet_codec_encoder_minimal.onnx"
    
    torch.onnx.export(
        encoder,
        dummy_input,
        str(onnx_path),
        input_names=['audio_padded'],
        output_names=['codes'],
        dynamic_axes={
            'audio_padded': {0: 'batch', 2: 'samples'},
            'codes': {0: 'batch', 2: 'time'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✓ Exported to: {onnx_path}")
    
    # Test ONNX model
    print("\nTesting ONNX model:")
    session = ort.InferenceSession(str(onnx_path))
    
    success_count = 0
    total_tests = 0
    
    for seconds in [0.5, 1, 2, 3, 5, 10]:
        samples = int(seconds * 44100)
        padded_samples = ((samples + 767) // 768) * 768
        
        audio_np = np.random.randn(1, 1, padded_samples).astype(np.float32)
        codes = session.run(None, {'audio_padded': audio_np})[0]
        
        expected_tokens = padded_samples // 768
        is_correct = codes.shape[2] == expected_tokens
        status = "✓" if is_correct else "✗"
        
        print(f"  {seconds:4.1f}s ({padded_samples:6d} samples): {codes.shape[2]:3d} tokens (expected {expected_tokens:3d}) {status}")
        
        if is_correct:
            success_count += 1
        total_tests += 1
    
    print(f"\nSuccess rate: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 SUCCESS! The minimal encoder works correctly!")
        print("\nIMPORTANT: This encoder requires pre-padded input.")
        print("Before encoding, pad your audio to a multiple of 768 samples.")
        
        # Save a helper function
        helper_code = '''def pad_audio_for_encoder(audio_np, hop_length=768):
    """Pad audio to multiple of hop_length for the minimal encoder."""
    samples = audio_np.shape[-1]
    padded_samples = ((samples + hop_length - 1) // hop_length) * hop_length
    pad_amount = padded_samples - samples
    
    if audio_np.ndim == 1:
        return np.pad(audio_np, (0, pad_amount), mode='constant')
    elif audio_np.ndim == 2:
        return np.pad(audio_np, ((0, 0), (0, pad_amount)), mode='constant')
    else:  # 3D
        return np.pad(audio_np, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
'''
        
        print("\nHelper function for padding:")
        print(helper_code)
    
    return onnx_path


if __name__ == "__main__":
    export_minimal()