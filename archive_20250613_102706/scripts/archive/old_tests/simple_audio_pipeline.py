"""
Simplified audio generation pipeline for testing ONNX models.
"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class SimpleVampNetPipeline:
    """Simplified pipeline for audio generation with ONNX models."""
    
    def __init__(self, transformer_path, codec_encoder_path, codec_decoder_path):
        """Initialize with ONNX model paths."""
        
        print("Loading ONNX models...")
        self.transformer = ort.InferenceSession(transformer_path)
        self.encoder = ort.InferenceSession(codec_encoder_path)
        self.decoder = ort.InferenceSession(codec_decoder_path)
        print("✓ All models loaded")
        
        # Model info
        self.n_coarse_codebooks = 4  # Transformer only handles coarse
        self.n_fine_codebooks = 10
        self.n_total_codebooks = 14  # LAC codec has 14 total
        self.seq_len = 100  # Fixed sequence length for transformer
        self.sample_rate = 16000
        self.hop_length = 320  # From codec configuration
        
    def encode_audio(self, audio):
        """Encode audio to discrete codes."""
        
        # Ensure correct shape: [batch, channels, samples]
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            if audio.shape[0] != 1:
                audio = audio.reshape(1, *audio.shape)
            if audio.shape[1] != 1:
                # Convert to mono
                audio = np.mean(audio, axis=1, keepdims=True)
        
        # Normalize audio
        audio = audio.astype(np.float32)
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Encode
        outputs = self.encoder.run(None, {'audio': audio})
        codes = outputs[0]
        
        return codes
    
    def decode_codes(self, codes):
        """Decode discrete codes to audio."""
        
        # Ensure int64
        codes = codes.astype(np.int64)
        
        # If we only have coarse codes, pad with zeros for fine codes
        if codes.shape[1] == self.n_coarse_codebooks:
            # Create fine codes (zeros = use default embeddings)
            fine_codes = np.zeros((codes.shape[0], self.n_fine_codebooks, codes.shape[2]), dtype=np.int64)
            # Concatenate coarse and fine
            codes = np.concatenate([codes, fine_codes], axis=1)
        
        # Decode
        outputs = self.decoder.run(None, {'codes': codes})
        audio = outputs[0]
        
        return audio
    
    def generate(self, codes, mask, temperature=1.0):
        """Generate new codes using the transformer."""
        
        # Handle sequence length mismatch
        seq_len = codes.shape[2]
        
        if seq_len != self.seq_len:
            # Process in chunks or pad
            if seq_len > self.seq_len:
                # Process in chunks
                generated_chunks = []
                
                for i in range(0, seq_len, self.seq_len):
                    end_idx = min(i + self.seq_len, seq_len)
                    chunk_len = end_idx - i
                    
                    # Extract chunk
                    codes_chunk = codes[:, :, i:end_idx]
                    mask_chunk = mask[:, :, i:end_idx]
                    
                    # Pad if necessary
                    if chunk_len < self.seq_len:
                        codes_chunk = np.pad(
                            codes_chunk,
                            ((0, 0), (0, 0), (0, self.seq_len - chunk_len)),
                            mode='constant',
                            constant_values=1024  # mask token
                        )
                        mask_chunk = np.pad(
                            mask_chunk,
                            ((0, 0), (0, 0), (0, self.seq_len - chunk_len)),
                            mode='constant',
                            constant_values=0
                        )
                    
                    # Generate
                    outputs = self.transformer.run(None, {
                        'codes': codes_chunk.astype(np.int64),
                        'mask': mask_chunk.astype(np.int64)
                    })
                    
                    # Extract valid part
                    generated_chunk = outputs[0][:, :, :chunk_len]
                    generated_chunks.append(generated_chunk)
                
                generated = np.concatenate(generated_chunks, axis=2)
                
            else:
                # Pad to expected length
                codes_padded = np.pad(
                    codes,
                    ((0, 0), (0, 0), (0, self.seq_len - seq_len)),
                    mode='constant',
                    constant_values=1024
                )
                mask_padded = np.pad(
                    mask,
                    ((0, 0), (0, 0), (0, self.seq_len - seq_len)),
                    mode='constant',
                    constant_values=0
                )
                
                outputs = self.transformer.run(None, {
                    'codes': codes_padded.astype(np.int64),
                    'mask': mask_padded.astype(np.int64)
                })
                
                generated = outputs[0][:, :, :seq_len]
        else:
            # Direct generation
            outputs = self.transformer.run(None, {
                'codes': codes.astype(np.int64),
                'mask': mask.astype(np.int64)
            })
            generated = outputs[0]
        
        return generated


def test_simple_pipeline():
    """Test the simplified pipeline."""
    
    print("=== Testing Simple Audio Pipeline ===\n")
    
    # Initialize pipeline
    # Try to find codec models
    encoder_paths = [
        "../models/codec_encoder.onnx",
        "../onnx_models_test/codec_encoder.onnx",
        "../onnx_models_optimized/codec_encoder.onnx",
        "../onnx_models/codec_encoder.onnx"
    ]
    decoder_paths = [
        "../models/codec_decoder.onnx",
        "../onnx_models_test/codec_decoder.onnx",
        "../onnx_models_optimized/codec_decoder.onnx",
        "../onnx_models/codec_decoder.onnx"
    ]
    
    encoder_path = None
    decoder_path = None
    
    for path in encoder_paths:
        if Path(path).exists():
            encoder_path = path
            break
    
    for path in decoder_paths:
        if Path(path).exists():
            decoder_path = path
            break
    
    if not encoder_path or not decoder_path:
        print("❌ Codec models not found!")
        print("Please export codec models first.")
        return
    
    print(f"Using encoder: {encoder_path}")
    print(f"Using decoder: {decoder_path}")
    
    pipeline = SimpleVampNetPipeline(
        transformer_path="vampnet_transformer_final.onnx",
        codec_encoder_path=encoder_path,
        codec_decoder_path=decoder_path
    )
    
    # Test 1: Generate from random codes
    print("\n1. Testing generation from random codes...")
    codes = np.random.randint(0, 1024, (1, pipeline.n_coarse_codebooks, 100), dtype=np.int64)
    mask = np.ones_like(codes)  # Mask everything
    
    generated = pipeline.generate(codes, mask)
    print(f"Generated shape: {generated.shape}")
    
    # Decode to audio
    audio = pipeline.decode_codes(generated)
    print(f"Decoded audio shape: {audio.shape}")
    
    # Save
    sf.write("test_simple_random.wav", audio[0].T, pipeline.sample_rate)
    print("✓ Saved to test_simple_random.wav")
    
    # Test 2: Load and regenerate audio
    print("\n2. Testing audio reconstruction...")
    
    # Create test audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration * pipeline.sample_rate))
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    
    # Encode
    codes = pipeline.encode_audio(test_audio)
    print(f"Encoded codes shape: {codes.shape}")
    
    # Handle sequence length mismatch
    seq_len = codes.shape[2]
    if seq_len < 100:
        # Pad to 100
        pad_len = 100 - seq_len
        codes = np.pad(codes, ((0, 0), (0, 0), (0, pad_len)), mode='constant', constant_values=0)
        print(f"Padded codes to shape: {codes.shape}")
    elif seq_len > 100:
        # Truncate to 100
        codes = codes[:, :, :100]
        print(f"Truncated codes to shape: {codes.shape}")
    
    # Partial mask
    mask = np.zeros_like(codes)
    mask[:, :, 30:min(70, seq_len)] = 1  # Mask middle section (up to original length)
    
    # Generate (transformer only handles coarse codes)
    coarse_codes = codes[:, :pipeline.n_coarse_codebooks, :]
    coarse_mask = mask[:, :pipeline.n_coarse_codebooks, :]
    generated_coarse = pipeline.generate(coarse_codes, coarse_mask)
    
    # Combine with fine codes
    generated = np.concatenate([generated_coarse, codes[:, pipeline.n_coarse_codebooks:, :]], axis=1)
    
    # If we padded, truncate back to original length for decoding
    if seq_len < 100:
        codes_to_decode = codes[:, :, :seq_len]
        generated_to_decode = generated[:, :, :seq_len]
    else:
        codes_to_decode = codes
        generated_to_decode = generated
    
    # Decode both
    original_audio = pipeline.decode_codes(codes_to_decode)
    generated_audio = pipeline.decode_codes(generated_to_decode)
    
    # Save
    sf.write("test_simple_original.wav", original_audio[0].T, pipeline.sample_rate)
    sf.write("test_simple_generated.wav", generated_audio[0].T, pipeline.sample_rate)
    print("✓ Saved original and generated audio")
    
    # Test 3: Iterative refinement
    print("\n3. Testing iterative refinement...")
    codes = np.random.randint(0, 1024, (1, pipeline.n_coarse_codebooks, 100), dtype=np.int64)
    
    for i in range(3):
        mask_ratio = 0.5 * (1 - i / 3)
        mask = np.zeros_like(codes)
        
        # Random masking
        for cb in range(pipeline.n_coarse_codebooks):
            positions = np.random.choice(100, int(100 * mask_ratio), replace=False)
            mask[0, cb, positions] = 1
        
        codes = pipeline.generate(codes, mask)
        print(f"  Step {i+1}: Masked {mask.sum()} positions")
    
    # Final decode
    final_audio = pipeline.decode_codes(codes)
    sf.write("test_simple_refined.wav", final_audio[0].T, pipeline.sample_rate)
    print("✓ Saved refined audio")
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_simple_pipeline()