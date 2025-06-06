"""
End-to-end VampNet ONNX pipeline.
"""

import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .validation import create_onnx_session, run_onnx_inference


class VampNetONNXPipeline:
    """
    Complete VampNet pipeline using ONNX models.
    """
    
    def __init__(self,
                 model_dir: str = "onnx_models",
                 providers: Optional[List[str]] = None):
        """
        Initialize pipeline with ONNX models.
        
        Args:
            model_dir: Directory containing ONNX models
            providers: ONNX Runtime providers
        """
        self.model_dir = Path(model_dir)
        self.providers = providers
        self.mask_token = 1024  # Default mask token
        
        # Load all models
        self.sessions = {}
        self._load_models()
        
    def _load_models(self):
        """Load all ONNX models."""
        model_files = {
            'audio_processor': 'audio_processor.onnx',
            'codec_encoder': 'codec_encoder.onnx',
            'mask_generator': 'mask_generator.onnx',
            'transformer': 'transformer.onnx',
            'codec_decoder': 'codec_decoder.onnx'
        }
        
        for name, filename in model_files.items():
            path = self.model_dir / filename
            if path.exists():
                print(f"Loading {name}...")
                self.sessions[name] = create_onnx_session(
                    str(path), 
                    providers=self.providers
                )
            else:
                print(f"Warning: {name} not found at {path}")
                
    def process_audio(self,
                     audio: np.ndarray,
                     sample_rate: int = 44100,
                     periodic_prompt: int = 7,
                     upper_codebook_mask: int = 3,
                     temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio: Input audio array [channels, samples]
            sample_rate: Audio sample rate
            periodic_prompt: Periodic masking interval
            upper_codebook_mask: Number of codebooks to mask
            temperature: Generation temperature
            
        Returns:
            Dictionary with intermediate and final results
        """
        results = {}
        
        # Add batch dimension if needed
        if audio.ndim == 2:
            audio = audio[np.newaxis, ...]  # [1, channels, samples]
        
        # Step 1: Audio preprocessing
        if 'audio_processor' in self.sessions:
            print("Processing audio...")
            processed = run_onnx_inference(
                self.sessions['audio_processor'],
                {'audio': audio}
            )
            processed_audio = processed['processed_audio']
            results['processed_audio'] = processed_audio
        else:
            # Fallback: simple preprocessing
            processed_audio = audio
            if processed_audio.shape[1] > 1:
                processed_audio = np.mean(processed_audio, axis=1, keepdims=True)
            results['processed_audio'] = processed_audio
            
        # Step 2: Encode to tokens
        if 'codec_encoder' in self.sessions:
            print("Encoding to tokens...")
            encoded = run_onnx_inference(
                self.sessions['codec_encoder'],
                {'audio': processed_audio}
            )
            codes = encoded['codes']
            results['codes'] = codes
        else:
            # Fallback: generate dummy tokens
            seq_len = processed_audio.shape[-1] // 768  # Assuming hop_length=768
            codes = np.random.randint(0, 1024, (1, 14, seq_len), dtype=np.int64)
            results['codes'] = codes
            
        # Step 3: Generate mask
        if 'mask_generator' in self.sessions:
            print("Generating mask...")
            mask_output = run_onnx_inference(
                self.sessions['mask_generator'],
                {'codes': codes}
            )
            mask = mask_output['mask']
            masked_codes = mask_output['masked_codes']
            results['mask'] = mask
            results['masked_codes'] = masked_codes
        else:
            # Fallback: simple periodic mask
            mask = np.ones_like(codes, dtype=np.int64)
            for i in range(0, codes.shape[-1], periodic_prompt):
                mask[:, :upper_codebook_mask, i] = 0
            mask[:, upper_codebook_mask:, :] = 1
            
            masked_codes = np.where(mask, 1024, codes)  # 1024 is mask token
            results['mask'] = mask
            results['masked_codes'] = masked_codes
            
        # Step 4: Generate new tokens
        if 'transformer' in self.sessions:
            print("Generating new tokens...")
            
            # Handle dynamic or fixed sequence length
            seq_len = masked_codes.shape[-1]
            
            # For transformer models, we need to handle fixed sequence lengths
            # Even if the model claims to support dynamic shapes, the internal reshapes might be fixed
            # We'll use a known working sequence length of 100 for this model
            expected_seq_len = 100  # This is what the model was exported with
            
            if seq_len != expected_seq_len:
                print(f"  Model expects sequence length {expected_seq_len}, got {seq_len}")
                
                if seq_len > expected_seq_len:
                    # Process in chunks
                    print(f"  Processing in chunks of {expected_seq_len}...")
                    generated_chunks = []
                    
                    for i in range(0, seq_len, expected_seq_len):
                        end_idx = min(i + expected_seq_len, seq_len)
                        chunk_len = end_idx - i
                        
                        # Pad if necessary
                        if chunk_len < expected_seq_len:
                            codes_chunk = np.pad(
                                masked_codes[:, :4, i:end_idx],
                                ((0, 0), (0, 0), (0, expected_seq_len - chunk_len)),
                                mode='constant',
                                constant_values=self.mask_token
                            )
                            mask_chunk = np.pad(
                                mask[:, :4, i:end_idx],
                                ((0, 0), (0, 0), (0, expected_seq_len - chunk_len)),
                                mode='constant',
                                constant_values=1
                            )
                        else:
                            codes_chunk = masked_codes[:, :4, i:end_idx]
                            mask_chunk = mask[:, :4, i:end_idx]
                        
                        # Generate for this chunk
                        chunk_result = run_onnx_inference(
                            self.sessions['transformer'],
                            {
                                'codes': codes_chunk,
                                'mask': mask_chunk
                            }
                        )
                        
                        # Extract only the valid part
                        generated_chunk = chunk_result['generated_codes'][:, :, :chunk_len]
                        generated_chunks.append(generated_chunk)
                    
                    # Concatenate all chunks
                    generated_codes = np.concatenate(generated_chunks, axis=-1)
                    
                elif seq_len < expected_seq_len:
                    # Pad to expected length
                    print(f"  Padding sequence from {seq_len} to {expected_seq_len}...")
                    codes_padded = np.pad(
                        masked_codes[:, :4, :],
                        ((0, 0), (0, 0), (0, expected_seq_len - seq_len)),
                        mode='constant',
                        constant_values=self.mask_token
                    )
                    mask_padded = np.pad(
                        mask[:, :4, :],
                        ((0, 0), (0, 0), (0, expected_seq_len - seq_len)),
                        mode='constant',
                        constant_values=1
                    )
                    
                    generated = run_onnx_inference(
                        self.sessions['transformer'],
                        {
                            'codes': codes_padded,
                            'mask': mask_padded
                        }
                    )
                    # Extract only the valid part
                    generated_codes = generated['generated_codes'][:, :, :seq_len]
                else:
                    # Exact match
                    generated = run_onnx_inference(
                        self.sessions['transformer'],
                        {
                            'codes': masked_codes[:, :4, :],
                            'mask': mask[:, :4, :]
                        }
                    )
                    generated_codes = generated['generated_codes']
            else:
                # Sequence length matches expected
                generated = run_onnx_inference(
                    self.sessions['transformer'],
                    {
                        'codes': masked_codes[:, :4, :],
                        'mask': mask[:, :4, :]
                    }
                )
                generated_codes = generated['generated_codes']
            
            # Combine with fine codes
            if codes.shape[1] > 4:
                full_codes = np.concatenate([
                    generated_codes,
                    codes[:, 4:, :]
                ], axis=1)
            else:
                full_codes = generated_codes
                
            results['generated_codes'] = full_codes
        else:
            # Fallback: return masked codes with some modifications
            generated_codes = masked_codes.copy()
            # Simulate generation by modifying masked positions
            generated_codes[mask == 1] = np.random.randint(0, 1024, np.sum(mask == 1))
            results['generated_codes'] = generated_codes
            
        # Step 5: Decode to audio
        if 'codec_decoder' in self.sessions:
            print("Decoding to audio...")
            decoded = run_onnx_inference(
                self.sessions['codec_decoder'],
                {'codes': results.get('generated_codes', codes)}
            )
            output_audio = decoded['audio']
            results['output_audio'] = output_audio
        else:
            # Fallback: generate dummy audio
            seq_len = results.get('generated_codes', codes).shape[-1]
            samples = seq_len * 768  # Assuming hop_length=768
            output_audio = np.random.randn(1, 1, samples) * 0.1
            results['output_audio'] = output_audio
            
        print("Pipeline complete!")
        return results
    
    def process_codes(self,
                     codes: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     periodic_prompt: int = 7,
                     upper_codebook_mask: int = 3) -> np.ndarray:
        """
        Process pre-encoded tokens (skip audio encoding/decoding).
        
        Args:
            codes: Input token codes [batch, n_codebooks, seq_len]
            mask: Optional pre-computed mask
            periodic_prompt: Periodic masking interval
            upper_codebook_mask: Number of codebooks to mask
            
        Returns:
            Generated token codes
        """
        # Generate mask if not provided
        if mask is None and 'mask_generator' in self.sessions:
            mask_output = run_onnx_inference(
                self.sessions['mask_generator'],
                {'codes': codes}
            )
            mask = mask_output['mask']
            masked_codes = mask_output['masked_codes']
        else:
            # Fallback mask
            if mask is None:
                mask = np.ones_like(codes, dtype=np.int64)
                for i in range(0, codes.shape[-1], periodic_prompt):
                    mask[:, :upper_codebook_mask, i] = 0
                mask[:, upper_codebook_mask:, :] = 1
            
            masked_codes = np.where(mask, 1024, codes)
            
        # Generate new tokens
        if 'transformer' in self.sessions:
            generated = run_onnx_inference(
                self.sessions['transformer'],
                {
                    'codes': masked_codes[:, :4, :],
                    'mask': mask[:, :4, :]
                }
            )
            generated_codes = generated['generated_codes']
            
            # Combine with fine codes
            if codes.shape[1] > 4:
                full_codes = np.concatenate([
                    generated_codes,
                    codes[:, 4:, :]
                ], axis=1)
            else:
                full_codes = generated_codes
                
            return full_codes
        else:
            # Fallback
            return masked_codes
            
    def warmup(self):
        """Warmup all models with dummy inputs."""
        print("Warming up models...")
        
        # Dummy inputs
        dummy_audio = np.random.randn(1, 2, 44100).astype(np.float32)
        dummy_codes = np.random.randint(0, 1024, (1, 14, 100), dtype=np.int64)
        dummy_mask = np.ones((1, 14, 100), dtype=np.int64)
        
        # Run each model
        if 'audio_processor' in self.sessions:
            try:
                _ = run_onnx_inference(
                    self.sessions['audio_processor'],
                    {'audio': dummy_audio}
                )
            except:
                pass
                
        if 'codec_encoder' in self.sessions:
            try:
                _ = run_onnx_inference(
                    self.sessions['codec_encoder'],
                    {'audio': dummy_audio[:, :1, :]}
                )
            except:
                pass
                
        if 'mask_generator' in self.sessions:
            try:
                _ = run_onnx_inference(
                    self.sessions['mask_generator'],
                    {'codes': dummy_codes}
                )
            except:
                pass
                
        if 'transformer' in self.sessions:
            try:
                _ = run_onnx_inference(
                    self.sessions['transformer'],
                    {
                        'codes': dummy_codes[:, :4, :],
                        'mask': dummy_mask[:, :4, :]
                    }
                )
            except:
                pass
                
        if 'codec_decoder' in self.sessions:
            try:
                _ = run_onnx_inference(
                    self.sessions['codec_decoder'],
                    {'codes': dummy_codes}
                )
            except:
                pass
                
        print("Warmup complete!")
        
    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about loaded models."""
        info = {}
        
        for name, session in self.sessions.items():
            model_info = {
                'inputs': {},
                'outputs': {}
            }
            
            # Get input info
            for inp in session.get_inputs():
                model_info['inputs'][inp.name] = {
                    'shape': inp.shape,
                    'type': inp.type
                }
                
            # Get output info
            for out in session.get_outputs():
                model_info['outputs'][out.name] = {
                    'shape': out.shape,
                    'type': out.type
                }
                
            info[name] = model_info
            
        return info