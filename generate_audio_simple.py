#!/usr/bin/env python3
"""
Simple audio generation using VampNet ONNX models.

This is a minimal example showing how to use the exported ONNX models.
"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path


def generate_audio_onnx(
    audio_path: str,
    output_path: str = "generated.wav",
    mask_ratio: float = 0.7,
    sample_rate: int = 44100
):
    """Generate audio using ONNX models.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save generated audio
        mask_ratio: Ratio of tokens to mask/generate
        sample_rate: Sample rate (default 44100)
    """
    
    # Load models
    print("Loading ONNX models...")
    encoder = ort.InferenceSession("scripts/models/vampnet_encoder_prepadded.onnx")
    decoder = ort.InferenceSession("scripts/models/vampnet_codec_decoder.onnx")
    coarse = ort.InferenceSession("onnx_models_fixed/coarse_complete_v3.onnx")
    c2f = ort.InferenceSession("onnx_models_fixed/c2f_complete_v3.onnx")
    
    # Load and prepare audio
    print(f"Loading audio from {audio_path}...")
    audio, sr = sf.read(audio_path)
    if sr != sample_rate:
        raise ValueError(f"Audio must be {sample_rate}Hz, got {sr}Hz")
    
    # Pad audio for encoder
    audio = audio.astype(np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, np.newaxis, :]
    
    hop_length = 768
    samples = audio.shape[-1]
    padded_samples = ((samples + hop_length - 1) // hop_length) * hop_length
    if padded_samples > samples:
        audio = np.pad(audio, ((0, 0), (0, 0), (0, padded_samples - samples)))
    
    # Encode
    print("Encoding audio...")
    codes = encoder.run(None, {'audio_padded': audio})[0]
    codes = codes[:, :, :100]  # Fixed length for ONNX
    
    # Generate coarse
    print("Generating coarse tokens...")
    coarse_codes = codes[:, :4, :].copy()
    mask = np.random.random((1, 4, 100)) < mask_ratio
    coarse_out = coarse.run(None, {
        'codes': coarse_codes.astype(np.int64),
        'mask': mask.astype(bool)
    })[0]
    
    # Generate fine
    print("Generating fine tokens...")
    c2f_input = np.zeros((1, 14, 100), dtype=np.int64)
    c2f_input[:, :4, :] = coarse_out
    c2f_mask = np.zeros((1, 14, 100), dtype=bool)
    c2f_mask[:, 4:, :] = True
    
    complete = c2f.run(None, {
        'codes': c2f_input,
        'mask': c2f_mask
    })[0]
    
    # Decode
    print("Decoding audio...")
    complete = np.clip(complete, 0, 1023)  # Fix any mask tokens
    audio_out = decoder.run(None, {'codes': complete.astype(np.int64)})[0]
    
    # Save
    audio_out = audio_out.squeeze()
    sf.write(output_path, audio_out, sample_rate)
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    generate_audio_onnx(
        "assets/stargazing.wav",
        "output_generated.wav",
        mask_ratio=0.7
    )