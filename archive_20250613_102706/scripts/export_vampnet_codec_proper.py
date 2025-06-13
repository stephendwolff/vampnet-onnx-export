#!/usr/bin/env python3
"""
Export the actual VampNet codec models to ONNX format.
This uses the real pretrained VampNet codec, not simplified placeholders.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet_onnx.vampnet_codec import VampNetCodecEncoder, VampNetCodecDecoder
import vampnet


def export_vampnet_codec():
    """Export the actual VampNet codec to ONNX."""
    print("=== Exporting VampNet Codec to ONNX ===\n")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Load VampNet interface to get the actual codec
    print("Loading VampNet codec from pretrained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load with specific paths to ensure we get the right model
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",  # Need this even though we only want codec
        wavebeat_ckpt="../models/vampnet/wavebeat.pth"
    )
    
    codec_model = interface.codec
    print(f"✓ Loaded codec: {codec_model}")
    print(f"  Sample rate: {codec_model.sample_rate}")
    print(f"  Hop length: {codec_model.hop_length}")
    print(f"  Codebooks: {codec_model.n_codebooks}")
    
    # Export encoder
    print("\n1. Exporting Encoder...")
    encoder_wrapper = VampNetCodecEncoder(codec_model, device=device)
    encoder_wrapper.eval()
    
    # Create example input - 3 seconds of audio
    example_audio = torch.randn(1, 1, 3 * 44100, device=device)
    
    encoder_path = output_dir / "vampnet_codec_encoder.onnx"
    
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
    test_audio = np.random.randn(1, 1, 44100).astype(np.float32)
    codes = encoder_session.run(None, {'audio': test_audio})[0]
    print(f"  Encoded shape: {codes.shape}")
    print(f"  Code range: [{codes.min()}, {codes.max()}]")
    print(f"  Unique values: {len(np.unique(codes))}")
    
    # Export decoder
    print("\n2. Exporting Decoder...")
    decoder_wrapper = VampNetCodecDecoder(codec_model, device=device)
    decoder_wrapper.eval()
    
    # Use the codes from encoder as example
    example_codes = torch.from_numpy(codes).long().to(device)
    
    decoder_path = output_dir / "vampnet_codec_decoder.onnx"
    
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
    reconstructed = decoder_session.run(None, {'codes': codes})[0]
    print(f"  Decoded shape: {reconstructed.shape}")
    print(f"  Audio range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Test round-trip
    print("\n3. Testing round-trip...")
    # Encode
    codes_rt = encoder_session.run(None, {'audio': test_audio})[0]
    # Decode
    audio_rt = decoder_session.run(None, {'codes': codes_rt})[0]
    
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