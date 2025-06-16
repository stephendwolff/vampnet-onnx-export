"""
Proper mask generator implementation that matches VampNet's behavior.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple


def _linear_random(shape: Tuple[int, ...], rand_mask_intensity: float = 1.0, 
                   seed: Optional[int] = None) -> np.ndarray:
    """Random masking with linear intensity."""
    if seed is not None:
        np.random.seed(seed)
    
    # VampNet uses rand_mask_intensity to control how much to mask
    # 1.0 = mask everything, 0.0 = mask nothing
    if rand_mask_intensity <= 0:
        return np.zeros(shape, dtype=bool)
    elif rand_mask_intensity >= 1:
        return np.ones(shape, dtype=bool)
    else:
        # Random mask with probability = rand_mask_intensity
        return np.random.rand(*shape) < rand_mask_intensity


def _inpaint(shape: Tuple[int, ...], prefix_s: float = 0.0, suffix_s: float = 0.0,
             hop_length: int = 768, sample_rate: int = 44100) -> np.ndarray:
    """Create inpainting mask that preserves prefix/suffix."""
    _, _, seq_len = shape
    mask = np.ones(shape, dtype=bool)
    
    # Convert seconds to token indices
    prefix_tokens = int(prefix_s * sample_rate / hop_length)
    suffix_tokens = int(suffix_s * sample_rate / hop_length)
    
    # Preserve prefix and suffix
    if prefix_tokens > 0:
        mask[:, :, :prefix_tokens] = False
    if suffix_tokens > 0:
        mask[:, :, -suffix_tokens:] = False
    
    return mask


def _periodic_mask(shape: Tuple[int, ...], periodic_prompt: int = 0, 
                   periodic_width: int = 1, random_roll: bool = False) -> np.ndarray:
    """Create periodic preservation mask."""
    if periodic_prompt <= 0:
        return np.ones(shape, dtype=bool)
    
    _, _, seq_len = shape
    mask = np.ones(shape, dtype=bool)
    
    # Create periodic pattern
    if random_roll:
        offset = np.random.randint(0, periodic_prompt)
    else:
        offset = 0
    
    for i in range(offset, seq_len, periodic_prompt):
        end_idx = min(i + periodic_width, seq_len)
        mask[:, :, i:end_idx] = False
    
    return mask


def _codebook_mask(shape: Tuple[int, ...], upper_codebook_mask: int = 0) -> np.ndarray:
    """Mask codebooks >= upper_codebook_mask completely."""
    mask = np.zeros(shape, dtype=bool)
    if upper_codebook_mask > 0:
        mask[:, upper_codebook_mask:, :] = True
    return mask


def build_mask_proper(
    z: np.ndarray,
    rand_mask_intensity: float = 1.0,
    prefix_s: float = 0.0,
    suffix_s: float = 0.0,
    periodic_prompt: int = 0,
    periodic_width: int = 1,
    upper_codebook_mask: int = 0,
    dropout: float = 0.0,
    random_roll: bool = False,
    seed: Optional[int] = None,
    hop_length: int = 768,
    sample_rate: int = 44100,
    **kwargs
) -> np.ndarray:
    """
    Build mask matching VampNet's masking behavior.
    
    Returns:
        Boolean mask where True = masked, False = preserved
    """
    shape = z.shape
    
    # Start with random masking
    mask = _linear_random(shape, rand_mask_intensity, seed)
    
    # Apply inpainting mask (AND operation - preserves prefix/suffix)
    if prefix_s > 0 or suffix_s > 0:
        inpaint_mask = _inpaint(shape, prefix_s, suffix_s, hop_length, sample_rate)
        mask = mask & inpaint_mask
    
    # Apply periodic mask (AND operation - preserves periodic positions)
    if periodic_prompt > 0:
        periodic = _periodic_mask(shape, periodic_prompt, periodic_width, random_roll)
        mask = mask & periodic
    
    # Apply dropout (additional random masking)
    if dropout > 0:
        dropout_mask = _linear_random(shape, dropout, seed)
        mask = mask | dropout_mask
    
    # Apply codebook mask (OR operation - always masks these codebooks)
    if upper_codebook_mask > 0:
        cb_mask = _codebook_mask(shape, upper_codebook_mask)
        mask = mask | cb_mask
    
    return mask.astype(bool)


class ProperMaskGenerator(nn.Module):
    """PyTorch version for ONNX export."""
    
    def __init__(self, hop_length: int = 768, sample_rate: int = 44100):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate
    
    def forward(
        self,
        z: torch.Tensor,
        rand_mask_intensity: float = 1.0,
        prefix_s: float = 0.0,
        suffix_s: float = 0.0,
        periodic_prompt: int = 0,
        periodic_width: int = 1,
        upper_codebook_mask: int = 0,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        """Generate mask in PyTorch for ONNX compatibility."""
        batch_size, n_codebooks, seq_len = z.shape
        device = z.device
        
        # Random mask
        if rand_mask_intensity <= 0:
            mask = torch.zeros_like(z, dtype=torch.bool)
        elif rand_mask_intensity >= 1:
            mask = torch.ones_like(z, dtype=torch.bool)
        else:
            mask = torch.rand_like(z, dtype=torch.float32) < rand_mask_intensity
        
        # Inpainting mask
        if prefix_s > 0 or suffix_s > 0:
            inpaint_mask = torch.ones_like(z, dtype=torch.bool)
            prefix_tokens = int(prefix_s * self.sample_rate / self.hop_length)
            suffix_tokens = int(suffix_s * self.sample_rate / self.hop_length)
            
            if prefix_tokens > 0:
                inpaint_mask[:, :, :prefix_tokens] = False
            if suffix_tokens > 0:
                inpaint_mask[:, :, -suffix_tokens:] = False
            
            mask = mask & inpaint_mask
        
        # Periodic mask
        if periodic_prompt > 0:
            periodic_mask = torch.ones_like(z, dtype=torch.bool)
            for i in range(0, seq_len, periodic_prompt):
                end_idx = min(i + periodic_width, seq_len)
                periodic_mask[:, :, i:end_idx] = False
            
            mask = mask & periodic_mask
        
        # Dropout
        if dropout > 0:
            dropout_mask = torch.rand_like(z, dtype=torch.float32) < dropout
            mask = mask | dropout_mask
        
        # Codebook mask
        if upper_codebook_mask > 0:
            cb_mask = torch.zeros_like(z, dtype=torch.bool)
            cb_mask[:, upper_codebook_mask:, :] = True
            mask = mask | cb_mask
        
        return mask