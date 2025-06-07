#!/usr/bin/env python3
"""
Example script showing how to use the VampNet ONNX pipeline for music generation.
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add parent directory to path to import vampnet_onnx
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import VampNetONNXPipeline


def load_audio(audio_path, sample_rate=44100):
    """Load audio file and convert to expected format."""
    audio, sr = sf.read(audio_path)
    
    # Resample if necessary (simple method, use librosa for better quality)
    if sr != sample_rate:
        print(f"Warning: Resampling from {sr}Hz to {sample_rate}Hz")
        # This is a simple nearest-neighbor resampling
        # For production, use librosa.resample
        ratio = sample_rate / sr
        new_length = int(len(audio) * ratio)
        indices = (np.arange(new_length) / ratio).astype(int)
        audio = audio[indices]
    
    # Convert to stereo if mono
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.T  # Transpose to [channels, samples]
    
    return audio, sample_rate


def generate_variations(input_audio, output_dir, pipeline, num_variations=3):
    """Generate multiple variations of the input audio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Different generation parameters
    configs = [
        {"name": "subtle", "periodic_prompt": 7, "upper_codebook_mask": 2, "mask_ratio": 0.3},
        {"name": "balanced", "periodic_prompt": 7, "upper_codebook_mask": 3, "mask_ratio": 0.5},
        {"name": "creative", "periodic_prompt": 10, "upper_codebook_mask": 4, "mask_ratio": 0.7},
    ]
    
    for i, config in enumerate(configs[:num_variations]):
        print(f"\nGenerating {config['name']} variation...")
        
        # Process audio
        results = pipeline.process_audio(
            input_audio,
            sample_rate=44100,
            periodic_prompt=config['periodic_prompt'],
            upper_codebook_mask=config['upper_codebook_mask'],
            mask_ratio=config.get('mask_ratio', 0.5)
        )
        
        # Save output
        if 'output_audio' in results:
            output_path = output_dir / f"variation_{i+1}_{config['name']}.wav"
            output_audio = results['output_audio'].squeeze()
            
            # Ensure audio is in correct shape for soundfile
            if output_audio.ndim == 1:
                sf.write(output_path, output_audio, 44100)
            else:
                sf.write(output_path, output_audio.T, 44100)
            
            print(f"Saved: {output_path}")
            
            # Print statistics
            print(f"  - Tokens processed: {results.get('num_tokens', 'N/A')}")
            print(f"  - Mask density: {results.get('mask_density', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Generate music variations using VampNet ONNX')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', default='generated', help='Output directory')
    parser.add_argument('-m', '--models', default='models', help='ONNX models directory')
    parser.add_argument('-n', '--num-variations', type=int, default=3, help='Number of variations')
    parser.add_argument('--max-duration', type=float, default=10.0, help='Maximum duration in seconds')
    
    args = parser.parse_args()
    
    # Load audio
    print(f"Loading audio from: {args.input}")
    audio, sr = load_audio(args.input)
    
    # Limit duration
    max_samples = int(args.max_duration * sr)
    if audio.shape[1] > max_samples:
        print(f"Limiting audio to {args.max_duration} seconds")
        audio = audio[:, :max_samples]
    
    # Initialize pipeline
    print(f"\nInitializing ONNX pipeline from: {args.models}")
    try:
        pipeline = VampNetONNXPipeline(model_dir=args.models)
        pipeline.warmup()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nMake sure you've exported the ONNX models first:")
        print("  python -m src.exporters --output-dir models")
        return 1
    
    # Generate variations
    generate_variations(audio, args.output, pipeline, args.num_variations)
    
    print(f"\nGeneration complete! Check {args.output}/ for results.")
    return 0


if __name__ == '__main__':
    sys.exit(main())