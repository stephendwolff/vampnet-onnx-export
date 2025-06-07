"""
Complete VampNet ONNX pipeline demo with both Coarse and C2F models.
This demonstrates the full two-stage generation process.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VampNetFullPipeline:
    """Complete VampNet pipeline using both coarse and C2F ONNX models."""
    
    def __init__(
        self,
        encoder_path: str = "models/codec_encoder.onnx",
        decoder_path: str = "models/codec_decoder.onnx",
        coarse_path: str = "onnx_models_fixed/coarse_transformer_v2_weighted.onnx",
        c2f_path: str = "onnx_models_fixed/c2f_transformer_v2_weighted.onnx"
    ):
        print("Initializing VampNet Full Pipeline...")
        
        # Check all models exist
        models = {
            "Encoder": encoder_path,
            "Decoder": decoder_path,
            "Coarse": coarse_path,
            "C2F": c2f_path
        }
        
        for name, path in models.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} model not found at {path}")
            print(f"✓ {name}: {path}")
        
        # Initialize ONNX sessions
        self.encoder_session = ort.InferenceSession(encoder_path)
        self.decoder_session = ort.InferenceSession(decoder_path)
        self.coarse_session = ort.InferenceSession(coarse_path)
        self.c2f_session = ort.InferenceSession(c2f_path)
        
        # Initialize processors
        # Note: These are PyTorch modules, not used directly in this demo
        
        # Model info
        self.sample_rate = 44100
        self.n_coarse_codebooks = 4
        self.n_fine_codebooks = 10
        self.total_codebooks = 14
        
        print("✓ Pipeline initialized successfully")
    
    def encode_audio(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Encode audio to latent codes using codec encoder."""
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # For now, skip audio processor and just prepare the audio
        # Add batch and channel dimensions
        if audio.ndim == 1:
            audio_input = audio[np.newaxis, np.newaxis, :]
        elif audio.ndim == 2:
            audio_input = audio[np.newaxis, :]
        else:
            audio_input = audio
        
        # Ensure float32
        audio_input = audio_input.astype(np.float32)
        
        # Encode
        encoder_outputs = self.encoder_session.run(None, {'audio': audio_input})
        codes = encoder_outputs[0]  # Shape: [batch, codebooks, time]
        
        return codes
    
    def generate_coarse(self, codes: np.ndarray, mask_ratio: float = 0.7) -> np.ndarray:
        """Generate coarse codes using the coarse transformer."""
        print(f"\nGenerating coarse codes (mask ratio: {mask_ratio})...")
        
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Use only coarse codebooks for input
        coarse_codes = codes[:, :self.n_coarse_codebooks, :]
        
        # Create mask for generation - simple random mask
        mask = np.random.random((batch_size, self.n_coarse_codebooks, seq_len)) < mask_ratio
        
        # Run coarse model
        # Check if model expects temperature input
        input_names = [i.name for i in self.coarse_session.get_inputs()]
        
        if 'temperature' in input_names:
            # Old model
            temperature = np.array(1.0, dtype=np.float32)
            inputs = {
                'codes': coarse_codes.astype(np.int64),
                'mask': mask.astype(bool),
                'temperature': temperature
            }
        else:
            # New v2 model
            inputs = {
                'codes': coarse_codes.astype(np.int64),
                'mask': mask.astype(bool)
            }
        
        coarse_outputs = self.coarse_session.run(None, inputs)
        
        generated_coarse = coarse_outputs[0]
        print(f"✓ Generated coarse codes shape: {generated_coarse.shape}")
        
        return generated_coarse
    
    def generate_fine(self, coarse_codes: np.ndarray) -> np.ndarray:
        """Generate fine codes using the C2F transformer."""
        print("\nGenerating fine codes...")
        
        batch_size, n_coarse, seq_len = coarse_codes.shape
        
        # Initialize fine codes with zeros (will be generated)
        fine_codes = np.zeros((batch_size, self.n_fine_codebooks, seq_len), dtype=np.int64)
        
        # Combine coarse (conditioning) and fine codes
        combined_codes = np.concatenate([coarse_codes, fine_codes], axis=1)
        
        # Create mask - mask all fine codes for generation
        mask = np.zeros((batch_size, self.total_codebooks, seq_len), dtype=bool)
        mask[:, self.n_coarse_codebooks:, :] = True  # Mask all fine codes
        
        # Run C2F model
        # Check if model expects temperature input
        input_names = [i.name for i in self.c2f_session.get_inputs()]
        
        if 'temperature' in input_names:
            # Old model
            temperature = np.array(1.0, dtype=np.float32)
            inputs = {
                'codes': combined_codes.astype(np.int64),
                'mask': mask.astype(bool),
                'temperature': temperature
            }
        else:
            # New v2 model
            inputs = {
                'codes': combined_codes.astype(np.int64),
                'mask': mask.astype(bool)
            }
        
        c2f_outputs = self.c2f_session.run(None, inputs)
        
        generated_codes = c2f_outputs[0]
        print(f"✓ Generated complete codes shape: {generated_codes.shape}")
        
        return generated_codes
    
    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Decode latent codes back to audio."""
        print("\nDecoding to audio...")
        
        # Ensure codes are int64
        codes = codes.astype(np.int64)
        
        # Handle mask tokens (1024) - decoder only accepts 0-1023
        # Replace mask tokens with 0 (silence/padding token)
        codes = np.where(codes == 1024, 0, codes)
        
        # Clip to valid range as extra safety
        codes = np.clip(codes, 0, 1023)
        
        # Run decoder
        decoder_outputs = self.decoder_session.run(None, {'codes': codes})
        audio = decoder_outputs[0]
        
        # Remove batch and channel dimensions
        audio = audio.squeeze()
        
        print(f"✓ Decoded audio shape: {audio.shape}")
        return audio
    
    def process_audio_full(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        mask_ratio: float = 0.7,
        save_intermediate: bool = False
    ) -> dict:
        """
        Process audio through the complete pipeline.
        
        Returns dict with:
        - original_audio: Input audio
        - coarse_audio: Audio from coarse codes only
        - final_audio: Audio from full pipeline
        - codes: Generated codes
        """
        results = {}
        
        # 1. Encode audio
        print("\n=== Stage 1: Encoding ===")
        original_codes = self.encode_audio(audio, sample_rate)
        results['original_codes'] = original_codes
        print(f"Encoded shape: {original_codes.shape}")
        
        # 2. Generate coarse codes
        print("\n=== Stage 2: Coarse Generation ===")
        coarse_codes = self.generate_coarse(original_codes, mask_ratio)
        results['coarse_codes'] = coarse_codes
        
        # Decode coarse-only (pad with zeros for fine codes)
        if save_intermediate:
            coarse_only_codes = np.zeros_like(original_codes)
            coarse_only_codes[:, :self.n_coarse_codebooks, :] = coarse_codes
            coarse_audio = self.decode_codes(coarse_only_codes)
            results['coarse_audio'] = coarse_audio
        
        # 3. Generate fine codes
        print("\n=== Stage 3: Fine Generation (C2F) ===")
        complete_codes = self.generate_fine(coarse_codes)
        results['complete_codes'] = complete_codes
        
        # 4. Decode final audio
        print("\n=== Stage 4: Final Decoding ===")
        final_audio = self.decode_codes(complete_codes)
        results['final_audio'] = final_audio
        
        return results


def demo_full_pipeline():
    """Demonstrate the full VampNet ONNX pipeline."""
    
    print("=== VampNet Full Pipeline Demo ===\n")
    
    # Initialize pipeline
    pipeline = VampNetFullPipeline()
    
    # Generate or load test audio
    print("\nGenerating test audio...")
    duration = 3.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more interesting test signal
    frequency1 = 440  # A4
    frequency2 = 554.37  # C#5
    audio = 0.3 * np.sin(2 * np.pi * frequency1 * t)
    audio += 0.2 * np.sin(2 * np.pi * frequency2 * t)
    audio += 0.1 * np.sin(2 * np.pi * frequency1 * 2 * t)  # Harmonic
    audio = audio.astype(np.float32)
    
    # Process through pipeline
    print("\nProcessing through full pipeline...")
    start_time = time.time()
    
    results = pipeline.process_audio_full(
        audio,
        sample_rate=sample_rate,
        mask_ratio=0.7,
        save_intermediate=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.2f} seconds")
    
    # Save outputs
    output_dir = Path("outputs/full_pipeline_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    sf.write(output_dir / "original.wav", audio, sample_rate)
    print(f"\nSaved original audio to {output_dir / 'original.wav'}")
    
    # Save coarse-only
    if 'coarse_audio' in results:
        sf.write(output_dir / "coarse_only.wav", results['coarse_audio'], sample_rate)
        print(f"Saved coarse-only audio to {output_dir / 'coarse_only.wav'}")
    
    # Save final
    sf.write(output_dir / "final_full_pipeline.wav", results['final_audio'], sample_rate)
    print(f"Saved final audio to {output_dir / 'final_full_pipeline.wav'}")
    
    # Save statistics
    stats = {
        'duration_seconds': duration,
        'sample_rate': sample_rate,
        'processing_time': elapsed,
        'original_shape': audio.shape,
        'coarse_codes_shape': results['coarse_codes'].shape,
        'complete_codes_shape': results['complete_codes'].shape,
        'final_audio_shape': results['final_audio'].shape,
        'pipeline_stages': [
            'Encode (Codec)',
            'Generate Coarse (Transformer)',
            'Generate Fine (C2F Transformer)',
            'Decode (Codec)'
        ]
    }
    
    with open(output_dir / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*50)
    print("Full pipeline demo completed!")
    print(f"Outputs saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    demo_full_pipeline()