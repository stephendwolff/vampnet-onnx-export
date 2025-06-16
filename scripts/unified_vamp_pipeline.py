#!/usr/bin/env python3
"""
Unified ONNX VampNet Pipeline.
Implements the complete vamp() method using ONNX models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import time
import soundfile as sf
from typing import Optional, Dict, Any
import onnxruntime as ort

sys.path.append(str(Path(__file__).parent.parent))

from scripts.iterative_generation import IterativeGenerator, create_onnx_generator
from vampnet.mask import linear_random, mask_and, inpaint, codebook_mask
import audiotools as at


class UnifiedVampPipeline:
    """
    Complete VampNet pipeline using ONNX models.
    Replicates the functionality of VampNet's Interface.vamp() method.
    """
    
    def __init__(self,
                 encoder_path: str = "scripts/models/vampnet_encoder_prepadded.onnx",
                 coarse_transformer_path: str = "vampnet_transformer_v11.onnx",
                 c2f_transformer_path: str = "vampnet_c2f_transformer_v15.onnx",
                 decoder_path: str = "scripts/models/vampnet_codec_decoder.onnx",
                 codec_ckpt_path: str = "models/vampnet/codec.pth",
                 device: str = "cpu"):
        
        self.device = device
        
        # Load ONNX models
        print("Loading ONNX models...")
        self.encoder = ort.InferenceSession(encoder_path) if Path(encoder_path).exists() else None
        self.decoder = ort.InferenceSession(decoder_path) if Path(decoder_path).exists() else None
        
        # Create iterative generators
        self.coarse_generator = create_onnx_generator(
            coarse_transformer_path,
            codec_ckpt_path,
            n_codebooks=4,
            latent_dim=8,
            mask_token=1024
        )
        
        if Path(c2f_transformer_path).exists():
            self.c2f_generator = create_onnx_generator(
                c2f_transformer_path,
                codec_ckpt_path,
                n_codebooks=14,
                latent_dim=8,
                mask_token=1024
            )
        else:
            self.c2f_generator = None
            print("Warning: C2F model not found, using coarse only")
        
        # Load codec info
        codec_ckpt = torch.load(codec_ckpt_path, map_location='cpu')
        self.sample_rate = 44100  # Default for LAC codec
        self.hop_length = 768
        
        print("✓ Pipeline initialized")
    
    def encode(self, audio_signal: at.AudioSignal) -> torch.Tensor:
        """Encode audio to tokens using ONNX encoder."""
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")
        
        # Preprocess audio
        signal = audio_signal.clone()
        signal = signal.resample(self.sample_rate).to_mono()
        signal = signal.normalize(-24.0).ensure_max_of_audio(1.0)
        
        # Pad to multiple of hop_length
        samples = signal.samples
        pad_length = (self.hop_length - (samples.shape[-1] % self.hop_length)) % self.hop_length
        if pad_length > 0:
            samples = F.pad(samples, (0, pad_length))
        
        # Run encoder
        codes = self.encoder.run(None, {'audio_padded': samples.numpy()})[0]
        codes = torch.from_numpy(codes).long()
        
        return codes
    
    def decode(self, codes: torch.Tensor) -> at.AudioSignal:
        """Decode tokens to audio using ONNX decoder."""
        if self.decoder is None:
            raise RuntimeError("Decoder not loaded")
        
        # Pad codes to 14 codebooks if needed
        if codes.shape[1] < 14:
            codes_padded = torch.zeros((codes.shape[0], 14, codes.shape[2]), dtype=torch.long)
            codes_padded[:, :codes.shape[1], :] = codes
            codes = codes_padded
        
        # Run decoder
        audio = self.decoder.run(None, {'codes': codes.numpy()})[0]
        
        # Create AudioSignal
        signal = at.AudioSignal(audio, self.sample_rate)
        
        return signal
    
    def coarse_vamp(self,
                    codes: torch.Tensor,
                    mask: torch.Tensor,
                    return_mask: bool = False,
                    time_steps: int = 12,
                    **kwargs) -> torch.Tensor:
        """
        Run coarse vamping - iterative generation on first 4 codebooks.
        """
        # Extract coarse codes
        coarse_codes = codes[:, :4, :].clone()
        coarse_mask = mask[:, :4, :]
        
        # Generate with coarse model
        generated = self.coarse_generator.generate(
            start_tokens=coarse_codes,
            mask=coarse_mask,
            time_steps=time_steps,
            **kwargs
        )
        
        # Create result with all 14 codebooks
        batch_size, _, seq_len = codes.shape
        result = torch.zeros((batch_size, 14, seq_len), dtype=torch.long)
        result[:, :4, :] = generated
        # Initialize fine codebooks with zeros (will be filled by C2F)
        
        if return_mask:
            return result, coarse_mask
        return result
    
    def coarse_to_fine(self,
                       codes: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       return_mask: bool = False,
                       chunk_size: int = 100,
                       **kwargs) -> torch.Tensor:
        """
        Run coarse-to-fine generation - add codebooks 5-14.
        """
        if self.c2f_generator is None:
            print("Warning: C2F not available, returning coarse codes")
            return codes
        
        batch_size, n_cb, seq_len = codes.shape
        
        # Ensure we have 14 codebooks
        if n_cb < 14:
            codes_full = torch.zeros((batch_size, 14, seq_len), dtype=torch.long)
            codes_full[:, :n_cb, :] = codes
            codes = codes_full
        
        if mask is not None and mask.shape[1] < 14:
            mask_full = torch.zeros((batch_size, 14, seq_len), dtype=torch.long)
            mask_full[:, :mask.shape[1], :] = mask
            mask = mask_full
        
        # Set mask for conditioning codebooks (first 4) to 0
        if mask is not None:
            mask = mask.clone()
            mask[:, :4, :] = 0
        else:
            mask = torch.ones_like(codes)
            mask[:, :4, :] = 0
        
        # Process in chunks
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        result_chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            
            chunk_codes = codes[:, :, start_idx:end_idx]
            chunk_mask = mask[:, :, start_idx:end_idx] if mask is not None else None
            
            # Generate fine codes
            chunk_result = self.c2f_generator.generate(
                start_tokens=chunk_codes,
                mask=chunk_mask,
                time_steps=2,  # C2F uses fewer steps
                **kwargs
            )
            
            result_chunks.append(chunk_result)
        
        # Concatenate chunks
        result = torch.cat(result_chunks, dim=2)
        
        if return_mask:
            return result[:, :, :seq_len], mask[:, :, :seq_len]
        return result[:, :, :seq_len]
    
    def vamp(self,
             codes: torch.Tensor,
             mask: torch.Tensor,
             batch_size: int = 1,
             feedback_steps: int = 1,
             return_mask: bool = False,
             **kwargs) -> torch.Tensor:
        """
        Complete vamp pipeline matching VampNet.Interface.vamp().
        """
        # Expand to batch size
        codes = codes.expand(batch_size, -1, -1)
        mask = mask.expand(batch_size, -1, -1)
        
        # Ensure mask covers all 14 codebooks
        if mask.shape[1] < 14:
            full_mask = torch.ones((batch_size, 14, mask.shape[2]), dtype=mask.dtype)
            full_mask[:, :mask.shape[1], :] = mask
            mask = full_mask
        
        # Coarse generation with feedback
        z = codes
        for i in range(feedback_steps):
            z = self.coarse_vamp(
                z,
                mask=mask,
                return_mask=False,
                **kwargs
            )
        
        # Coarse-to-fine
        if self.c2f_generator is not None:
            try:
                c2f_kwargs = kwargs.copy()
                c2f_kwargs['typical_filtering'] = True
                c2f_kwargs['temperature'] = kwargs.get('temperature', 1.0)
                z = self.coarse_to_fine(
                    z,
                    mask=mask,
                    **c2f_kwargs
                )
            except Exception as e:
                print(f"Warning: C2F failed ({e}), using coarse only")
                # Keep coarse result
        
        if return_mask:
            return z, mask
        return z
    
    def process_audio(self,
                      input_audio: at.AudioSignal,
                      mask_ratio: float = 0.8,
                      feedback_steps: int = 1,
                      **kwargs) -> at.AudioSignal:
        """
        Complete audio-to-audio vamping process.
        """
        print("\n1. Encoding audio...")
        codes = self.encode(input_audio)
        print(f"   Encoded shape: {codes.shape}")
        
        print("\n2. Creating mask...")
        mask = linear_random(codes, mask_ratio)
        mask = codebook_mask(mask, 3)  # Mask first 3 codebooks
        print(f"   Masked {mask.sum().item()} positions")
        
        print("\n3. Running vamp...")
        start_time = time.time()
        vamped_codes = self.vamp(
            codes,
            mask,
            feedback_steps=feedback_steps,
            **kwargs
        )
        vamp_time = time.time() - start_time
        print(f"   Vamp time: {vamp_time:.2f}s")
        
        print("\n4. Decoding audio...")
        output_audio = self.decode(vamped_codes)
        
        return output_audio


def test_unified_pipeline():
    """Test the unified ONNX vamp pipeline."""
    print("="*80)
    print("TESTING UNIFIED ONNX VAMP PIPELINE")
    print("="*80)
    
    # Create pipeline
    pipeline = UnifiedVampPipeline()
    
    # Create test audio
    print("\nCreating test audio...")
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Simple tone
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    test_signal = at.AudioSignal(audio[None, None, :], sample_rate)
    
    # Process
    print("\nProcessing audio through pipeline...")
    output_signal = pipeline.process_audio(
        test_signal,
        mask_ratio=0.7,
        temperature=1.0,
        top_k=50,
        typical_filtering=True,
        typical_mass=0.15
    )
    
    print(f"\nOutput shape: {output_signal.samples.shape}")
    print(f"Output duration: {output_signal.duration:.2f}s")
    
    # Save results
    test_signal.write("test_input_vamp.wav")
    output_signal.write("test_output_vamp.wav")
    print("\n✓ Saved test_input_vamp.wav and test_output_vamp.wav")
    
    # Test with real audio if available
    if Path("assets/example.wav").exists():
        print("\n\nTesting with real audio...")
        real_signal = at.AudioSignal("assets/example.wav")
        real_signal = real_signal[:, :5]  # First 5 seconds
        
        output_real = pipeline.process_audio(
            real_signal,
            mask_ratio=0.5,
            temperature=0.8,
            top_k=50
        )
        
        real_signal.write("real_input_vamp.wav")
        output_real.write("real_output_vamp.wav")
        print("✓ Saved real_input_vamp.wav and real_output_vamp.wav")
    
    print("\n✅ Unified pipeline test complete!")
    print("="*80)


if __name__ == "__main__":
    test_unified_pipeline()