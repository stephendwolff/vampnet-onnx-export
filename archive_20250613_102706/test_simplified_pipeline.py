#!/usr/bin/env python3
"""
Test the complete ONNX pipeline with simplified codec.
"""

import numpy as np
import soundfile as sf
from vampnet_onnx.pipeline import VampNetONNXPipeline
from vampnet_onnx.exporters import export_all_components
import os

def test_pipeline():
    # Export all components with simplified codec
    print("Exporting ONNX models...")
    exported_models = export_all_components(
        output_dir="models",
        codec_encoder={'use_simplified': True},
        codec_decoder={'use_simplified': True},
        transformer={'use_simplified': True, 'example_sequence_length': 100}
    )
    
    print("\nInitializing ONNX pipeline...")
    pipeline = VampNetONNXPipeline(model_dir="models")
    
    # Create test audio
    print("\nCreating test audio...")
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Simple melody
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
        0.2 * np.sin(2 * np.pi * 659.25 * t)   # E5
    )
    
    # Add stereo dimension and ensure float32
    test_audio = np.stack([test_audio, test_audio]).astype(np.float32)  # [2, samples]
    
    print(f"Test audio shape: {test_audio.shape}")
    print(f"Test audio dtype: {test_audio.dtype}")
    
    # Process through pipeline
    print("\nProcessing audio through pipeline...")
    results = pipeline.process_audio(
        test_audio,
        sample_rate=sample_rate,
        periodic_prompt=7,
        upper_codebook_mask=3
    )
    
    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    
    # Save original
    sf.write("outputs/pipeline_test_original.wav", test_audio.T, sample_rate)
    print("Saved original to outputs/pipeline_test_original.wav")
    
    # Save processed
    if 'output_audio' in results:
        output_audio = results['output_audio']
        print(f"Output shape: {output_audio.shape}")
        
        # Convert to proper shape for soundfile
        if output_audio.ndim == 3:  # [batch, channels, samples]
            output_audio = output_audio[0]  # Remove batch dimension
        if output_audio.shape[0] == 1:  # Mono
            output_audio = output_audio[0]
        else:  # Stereo
            output_audio = output_audio.T
            
        sf.write("outputs/pipeline_test_output.wav", output_audio, sample_rate)
        print("Saved output to outputs/pipeline_test_output.wav")
        
        # Print stats
        print(f"\nProcessing stats:")
        print(f"Original length: {test_audio.shape[1] / sample_rate:.2f} seconds")
        print(f"Output length: {len(output_audio) / sample_rate:.2f} seconds")
        if 'codes' in results:
            print(f"Token sequence length: {results['codes'].shape[-1]}")
    
    print("\nPipeline test complete!")
    
if __name__ == "__main__":
    test_pipeline()