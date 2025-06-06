"""
Audio preprocessing module for VampNet ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AudioProcessor(nn.Module):
    """
    Audio preprocessing module compatible with ONNX export.
    
    Handles:
    - Sample rate conversion (simplified for ONNX)
    - Mono conversion
    - Loudness normalization
    - Padding/trimming to fixed lengths
    """
    
    def __init__(self, 
                 target_sample_rate: int = 44100,
                 target_loudness: float = -24.0,
                 hop_length: int = 768):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.target_loudness = target_loudness
        self.hop_length = hop_length
        
        # Pre-compute normalization factor
        self.loudness_scale = 10 ** (target_loudness / 20)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio for VampNet encoding.
        
        Args:
            audio: Input audio [batch, channels, samples]
            
        Returns:
            Processed audio [batch, 1, samples]
        """
        batch_size = audio.shape[0]
        
        # Convert to mono using mean across channels
        # This works for both mono (no-op) and stereo without conditionals
        audio = audio.mean(dim=1, keepdim=True)
        
        # Compute RMS for loudness normalization (similar to audiotools)
        # First compute the RMS of the audio
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        rms = torch.clamp(rms, min=1e-8)  # Avoid division by zero
        
        # Convert target loudness from dB to linear scale
        # target_loudness_linear = 10^(target_loudness_db / 20)
        target_rms = 10 ** (self.target_loudness / 20)
        
        # Scale audio to match target RMS
        audio = audio * (target_rms / rms)
        
        # Ensure max value doesn't exceed 1.0 (matching original implementation)
        max_val = torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        # Only scale down if max > 1.0
        scale_factor = torch.where(max_val > 1.0, 1.0 / max_val, torch.ones_like(max_val))
        audio = audio * scale_factor
        
        # Pad to multiple of hop_length
        samples = audio.shape[-1]
        target_length = ((samples + self.hop_length - 1) // self.hop_length) * self.hop_length
        
        # Always pad (padding might be 0 which is fine)
        padding_amount = target_length - samples
        audio = F.pad(audio, (0, padding_amount))
            
        return audio
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output length after padding."""
        return ((input_length + self.hop_length - 1) // self.hop_length) * self.hop_length


class AudioPostProcessor(nn.Module):
    """
    Audio post-processing module for ONNX export.
    
    Handles:
    - Trimming to original length
    - Output normalization
    """
    
    def __init__(self, target_sample_rate: int = 44100):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        
    def forward(self, audio: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        Post-process audio after VampNet decoding.
        
        Args:
            audio: Decoded audio [batch, 1, samples]
            original_length: Original audio length before padding
            
        Returns:
            Trimmed audio [batch, 1, original_length]
        """
        # Trim to original length
        audio = audio[:, :, :original_length]
        
        # Ensure output is in valid range
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio