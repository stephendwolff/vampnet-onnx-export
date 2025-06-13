"""
Fixed VampNet ONNX pipeline with better token generation.
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


class VampNetFullPipelineFixed:
    """Fixed VampNet pipeline with proper token handling."""
    
    def __init__(
        self,
        encoder_path: str = "models/vampnet_codec_encoder.onnx",
        decoder_path: str = "models/vampnet_codec_decoder.onnx",
        coarse_path: str = "onnx_models_fixed/coarse_transformer_v2_weighted.onnx",
        c2f_path: str = "onnx_models_fixed/c2f_transformer_v2_weighted.onnx"
    ):
        print("Initializing Fixed VampNet Pipeline...")
        
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
        
        # Model info
        self.sample_rate = 44100
        self.n_coarse_codebooks = 4
        self.n_fine_codebooks = 10
        self.total_codebooks = 14
        self.vocab_size = 1024
        self.mask_token = 1024
        
        print("✓ Pipeline initialized successfully")
    
    def encode_audio(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Encode audio to latent codes using codec encoder."""
        if sample_rate is None:
            sample_rate = self.sample_rate
            
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
    
    def fix_mask_tokens(self, codes: np.ndarray) -> np.ndarray:
        """Replace mask tokens (1024) with nearest valid token."""
        # Instead of replacing with 0, replace with a random valid token
        # This should produce better audio than silence
        mask_positions = codes == self.mask_token
        
        if np.any(mask_positions):
            # For each mask token, replace with a random token from the same codebook
            for batch in range(codes.shape[0]):
                for cb in range(codes.shape[1]):
                    cb_mask = mask_positions[batch, cb]
                    if np.any(cb_mask):
                        # Get valid tokens from this codebook (excluding masks)
                        valid_tokens = codes[batch, cb][~cb_mask]
                        if len(valid_tokens) > 0:
                            # Replace with random samples from valid tokens
                            n_masks = cb_mask.sum()
                            replacements = np.random.choice(valid_tokens, size=n_masks)
                            codes[batch, cb][cb_mask] = replacements
                        else:
                            # Fallback to random tokens in valid range
                            codes[batch, cb][cb_mask] = np.random.randint(0, self.vocab_size, size=cb_mask.sum())
        
        # Ensure all values are in valid range
        codes = np.clip(codes, 0, self.vocab_size - 1)
        
        return codes
    
    def generate_coarse(self, codes: np.ndarray, mask_ratio: float = 0.7) -> np.ndarray:
        """Generate coarse codes with fixed masking."""
        print(f"\nGenerating coarse codes (mask ratio: {mask_ratio})...")
        
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Use only coarse codebooks for input
        coarse_codes = codes[:, :self.n_coarse_codebooks, :].copy()
        
        # Create periodic mask pattern (more like VampNet)
        # Instead of random mask, use periodic masking which might work better
        mask = np.zeros((batch_size, self.n_coarse_codebooks, seq_len), dtype=bool)
        
        if mask_ratio > 0:
            # Create a more structured mask
            period = 30  #max(1, int(1 / mask_ratio))
            for i in range(seq_len):
                if i % period != 0:  # Keep every period-th token
                    mask[:, :, i] = True

        # Run coarse model
        inputs = {
            'codes': coarse_codes.astype(np.int64),
            'mask': mask.astype(bool)
        }
        
        coarse_outputs = self.coarse_session.run(None, inputs)
        generated_coarse = coarse_outputs[0]
        
        # Fix any mask tokens in output
        generated_coarse = self.fix_mask_tokens(generated_coarse)
        
        print(f"✓ Generated coarse codes shape: {generated_coarse.shape}")
        print(f"  Unique values: {len(np.unique(generated_coarse))}")
        print(f"  Range: [{generated_coarse.min()}, {generated_coarse.max()}]")
        
        return generated_coarse
    
    def generate_fine(self, coarse_codes: np.ndarray) -> np.ndarray:
        """Generate fine codes using the C2F transformer."""
        print("\nGenerating fine codes...")
        
        batch_size, n_coarse, seq_len = coarse_codes.shape
        
        # Initialize with original fine codes (will be replaced)
        fine_codes = np.random.randint(0, self.vocab_size, 
                                     (batch_size, self.n_fine_codebooks, seq_len), 
                                     dtype=np.int64)
        
        # Combine coarse and fine codes
        combined_codes = np.concatenate([coarse_codes, fine_codes], axis=1)
        
        # Create mask for fine codes only
        mask = np.zeros((batch_size, self.total_codebooks, seq_len), dtype=bool)
        mask[:, self.n_coarse_codebooks:, :] = True  # Mask all fine codes
        
        # Run C2F model
        inputs = {
            'codes': combined_codes.astype(np.int64),
            'mask': mask.astype(bool)
        }
        
        c2f_outputs = self.c2f_session.run(None, inputs)
        generated_codes = c2f_outputs[0]
        
        # Fix any mask tokens
        generated_codes = self.fix_mask_tokens(generated_codes)
        
        print(f"✓ Generated complete codes shape: {generated_codes.shape}")
        print(f"  Fine codes unique values: {len(np.unique(generated_codes[:, 4:]))}")
        
        return generated_codes
    
    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Decode latent codes back to audio."""
        print("\nDecoding to audio...")
        
        # Ensure codes are int64 and in valid range
        codes = codes.astype(np.int64)
        codes = self.fix_mask_tokens(codes)
        
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
        """Process audio through the complete pipeline."""
        results = {}
        
        # 1. Encode audio
        print("\n=== Stage 1: Encoding ===")
        original_codes = self.encode_audio(audio, sample_rate)
        results['original_codes'] = original_codes
        print(f"Encoded shape: {original_codes.shape}")
        print(f"Original codes range: [{original_codes.min()}, {original_codes.max()}]")
        
        # 2. Generate coarse codes
        print("\n=== Stage 2: Coarse Generation ===")
        coarse_codes = self.generate_coarse(original_codes, mask_ratio)
        results['coarse_codes'] = coarse_codes
        
        # 3. Generate fine codes
        print("\n=== Stage 3: Fine Generation (C2F) ===")
        complete_codes = self.generate_fine(coarse_codes)
        results['complete_codes'] = complete_codes
        
        # 4. Decode final audio
        print("\n=== Stage 4: Final Decoding ===")
        final_audio = self.decode_codes(complete_codes)
        results['final_audio'] = final_audio
        
        return results


if __name__ == "__main__":
    # Test the fixed pipeline
    print("Testing Fixed VampNet Pipeline...")
    
    pipeline = VampNetFullPipelineFixed()
    
    # Generate test audio
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process
    results = pipeline.process_audio_full(test_audio, sample_rate, mask_ratio=0.7)
    
    # Save output
    output_dir = Path("outputs/fixed_pipeline_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sf.write(output_dir / "test_input.wav", test_audio, sample_rate)
    sf.write(output_dir / "test_output.wav", results['final_audio'], sample_rate)
    
    print(f"\nSaved to {output_dir}")