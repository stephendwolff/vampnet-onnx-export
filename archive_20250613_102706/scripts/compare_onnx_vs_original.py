"""
Compare ONNX models vs original VampNet to verify quality.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import vampnet
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vampnet_onnx.pipeline import VampNetONNXPipeline


def compare_models():
    """Compare original VampNet with ONNX models."""
    
    print("=== VampNet ONNX vs Original Comparison ===\n")
    
    # 1. Generate test audio
    print("Generating test audio...")
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 2. Process with original VampNet
    print("\n--- Processing with Original VampNet ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        interface = vampnet.interface.Interface(device=device)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(test_audio).unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            z = interface.encode(audio_tensor)
            print(f"Encoded shape: {z.shape}")
            
            # Apply some masking for generation
            mask = torch.ones_like(z).bool()
            mask[:, :, z.shape[2]//2:] = False  # Mask second half
            
            # Coarse generation
            z_coarse = z[:, :4].clone()
            z_coarse[~mask[:, :4]] = interface.coarse.mask_token
            z_coarse_gen = interface.coarse.generate(z_coarse, mask=mask[:, :4])
            
            # C2F generation
            z_full = torch.cat([z_coarse_gen, z[:, 4:]], dim=1)
            z_full[:, 4:][~mask[:, 4:]] = interface.c2f.mask_token
            z_full_gen = interface.c2f.generate(z_full, mask=mask)
            
            # Decode
            audio_vampnet = interface.decode(z_full_gen)
            audio_vampnet = audio_vampnet.squeeze().cpu().numpy()
            
        print(f"Generated audio shape: {audio_vampnet.shape}")
        
    except Exception as e:
        print(f"Error with original VampNet: {e}")
        audio_vampnet = None
    
    # 3. Process with ONNX pipeline
    print("\n--- Processing with ONNX Pipeline ---")
    
    pipeline = VampNetONNXPipeline(
        encoder_path="models/codec_encoder.onnx",
        decoder_path="models/codec_decoder.onnx",
        transformer_path="onnx_models_fixed/coarse_transformer_v2_weighted.onnx"
    )
    
    # Note: This pipeline only has coarse model, not full C2F
    codes = pipeline.encode_audio(test_audio, sample_rate)
    print(f"Encoded shape: {codes.shape}")
    
    # Decode back
    audio_onnx = pipeline.decode_codes(codes)
    print(f"Generated audio shape: {audio_onnx.shape}")
    
    # 4. Save outputs
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sf.write(output_dir / "test_input.wav", test_audio, sample_rate)
    print(f"\nSaved test input to {output_dir / 'test_input.wav'}")
    
    if audio_vampnet is not None:
        sf.write(output_dir / "vampnet_original.wav", audio_vampnet, sample_rate)
        print(f"Saved VampNet output to {output_dir / 'vampnet_original.wav'}")
    
    sf.write(output_dir / "vampnet_onnx.wav", audio_onnx, sample_rate)
    print(f"Saved ONNX output to {output_dir / 'vampnet_onnx.wav'}")
    
    # 5. Compare metrics
    if audio_vampnet is not None:
        min_len = min(len(audio_vampnet), len(audio_onnx))
        mse = np.mean((audio_vampnet[:min_len] - audio_onnx[:min_len])**2)
        print(f"\nMSE between outputs: {mse:.6f}")
    
    print("\n" + "="*60)
    print("Comparison complete! Check the output files to compare quality.")
    print("Note: ONNX pipeline currently only uses coarse model.")
    print("For full quality, use the full pipeline demo with both models.")
    print("="*60)


if __name__ == "__main__":
    compare_models()