"""
Test ONNX model with real audio generation.

This script loads an audio file, encodes it with the codec,
generates tokens using the ONNX model, and decodes back to audio.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vampnet_onnx.audio_processor import AudioProcessor
from vampnet_onnx.vampnet_codec import VampNetCodec
from vampnet_onnx.mask_generator_onnx import MaskGenerator


def generate_with_onnx_model(
    audio_path: str,
    onnx_model_path: str = "onnx_models_fixed/coarse_complete_v3.onnx",
    output_path: str = "outputs/onnx_generated.wav",
    mask_ratio: float = 0.5,
    sampling_steps: int = 12,
):
    """Generate audio using ONNX model."""
    
    print("=== Testing ONNX Audio Generation ===")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\nInitializing components...")
    audio_processor = AudioProcessor(sample_rate=44100, target_sample_rate=44100)
    codec = VampNetCodec(
        encoder_path="models/vampnet_codec_encoder.onnx",
        decoder_path="models/vampnet_codec_decoder.onnx"
    )
    mask_generator = MaskGenerator()
    
    # Load ONNX model
    print(f"\nLoading ONNX model from {onnx_model_path}")
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Print model info
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    
    # Load and process audio
    print(f"\nLoading audio from {audio_path}")
    waveform, sample_rate = audio_processor.load_audio(audio_path)
    print(f"Original shape: {waveform.shape}, sample rate: {sample_rate}")
    
    # Resample if needed
    if sample_rate != 44100:
        waveform = audio_processor.resample(waveform, sample_rate, 44100)
        sample_rate = 44100
    
    # Normalize
    waveform = audio_processor.normalize(waveform)
    
    # Ensure correct shape for codec
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    print(f"Processed shape: {waveform.shape}")
    
    # Encode to tokens
    print("\nEncoding audio to tokens...")
    tokens = codec.encode(waveform.numpy())
    print(f"Token shape: {tokens.shape}")
    
    # Take only coarse codebooks (first 4)
    coarse_tokens = tokens[:, :4, :]
    print(f"Coarse token shape: {coarse_tokens.shape}")
    
    # Generate mask
    print(f"\nGenerating mask with ratio {mask_ratio}")
    mask = mask_generator.generate_mask(
        shape=coarse_tokens.shape,
        mask_ratio=mask_ratio,
        mask_mode="random"
    )
    print(f"Masked positions: {mask.sum()}/{mask.numel()} ({100*mask.sum()/mask.numel():.1f}%)")
    
    # Apply mask to tokens
    masked_tokens = coarse_tokens.copy()
    masked_tokens[mask] = 1024  # mask token
    
    # Iterative generation
    print(f"\nGenerating with {sampling_steps} steps...")
    
    for step in range(sampling_steps):
        # Run ONNX inference
        ort_inputs = {
            'codes': masked_tokens,
            'mask': mask
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        generated = ort_outputs[0]
        
        # Update masked tokens with generated values
        masked_tokens[mask] = generated[mask]
        
        # Update mask for next iteration (if not last step)
        if step < sampling_steps - 1:
            # Reduce mask ratio gradually
            current_ratio = mask_ratio * (1 - (step + 1) / sampling_steps)
            mask = mask_generator.generate_mask(
                shape=coarse_tokens.shape,
                mask_ratio=current_ratio,
                mask_mode="random"
            )
            # Ensure we don't unmask already generated tokens
            masked_tokens[mask] = 1024
        
        print(f"  Step {step+1}/{sampling_steps} - Unique tokens: {len(np.unique(generated))}")
    
    # Final generated tokens
    generated_tokens = masked_tokens
    print(f"\nFinal token shape: {generated_tokens.shape}")
    print(f"Token range: [{generated_tokens.min()}, {generated_tokens.max()}]")
    print(f"Unique tokens: {len(np.unique(generated_tokens))}")
    
    # For full decoding, we need all 14 codebooks
    # Duplicate coarse to create a full set (simplified approach)
    full_tokens = np.zeros((1, 14, generated_tokens.shape[2]), dtype=np.int64)
    
    # Copy coarse tokens
    full_tokens[:, :4, :] = generated_tokens
    
    # For fine tokens, we can either:
    # 1. Use original fine tokens if available
    # 2. Duplicate coarse patterns (simplified)
    # 3. Use zeros (will give lower quality)
    
    # For now, let's duplicate some coarse patterns
    for i in range(4, 14):
        full_tokens[:, i, :] = generated_tokens[:, i % 4, :]
    
    # Decode back to audio
    print("\nDecoding tokens to audio...")
    generated_waveform = codec.decode(full_tokens)
    generated_waveform = torch.from_numpy(generated_waveform)
    
    # Denormalize
    generated_waveform = audio_processor.denormalize(generated_waveform)
    
    # Save output
    print(f"\nSaving to {output_path}")
    audio_processor.save_audio(generated_waveform, output_path, sample_rate)
    
    print("\n✓ Generation complete!")
    
    # Compare with original
    print("\n=== Comparison ===")
    print(f"Original duration: {waveform.shape[-1] / sample_rate:.2f}s")
    print(f"Generated duration: {generated_waveform.shape[-1] / sample_rate:.2f}s")
    
    return generated_waveform


def compare_with_vampnet(audio_path: str):
    """Compare ONNX generation with original VampNet."""
    
    print("\n=== Comparing with VampNet ===")
    
    try:
        from vampnet import interface as vampnet
        
        # Load VampNet
        print("Loading VampNet interface...")
        interface = vampnet.Interface(
            coarse_ckpt="models/vampnet/coarse.pth",
            coarse2fine_ckpt="models/vampnet/c2f.pth",
            codec_ckpt="models/vampnet/codec.pth",
            device="cpu"
        )
        
        # Load audio
        import audiotools as at
        signal = interface.load_audio(audio_path)
        
        # Generate with VampNet
        print("Generating with VampNet...")
        vampnet_output = interface.generate(
            signal,
            mask_ratio=0.5,
            sampling_steps=12,
            temperature=1.0
        )
        
        # Save VampNet output
        vampnet_output.save("outputs/vampnet_generated.wav")
        print("✓ VampNet generation saved to outputs/vampnet_generated.wav")
        
    except Exception as e:
        print(f"Could not run VampNet comparison: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX audio generation")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="onnx_models_fixed/coarse_complete_v3.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/onnx_generated.wav",
        help="Path to output audio file",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Mask ratio for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with VampNet output",
    )
    
    args = parser.parse_args()
    
    # Generate with ONNX
    generated = generate_with_onnx_model(
        audio_path=args.audio_path,
        onnx_model_path=args.model,
        output_path=args.output,
        mask_ratio=args.mask_ratio,
        sampling_steps=args.steps,
    )
    
    # Compare with VampNet if requested
    if args.compare:
        compare_with_vampnet(args.audio_path)