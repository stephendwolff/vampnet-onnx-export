"""
ONNX-compatible mask generation module for VampNet.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ONNXMaskGenerator(nn.Module):
    """
    ONNX-compatible mask generator that avoids Python conditionals.
    Parameters are fixed at initialization time.
    """
    
    def __init__(self,
                 n_codebooks: int = 14,
                 mask_token: int = 1024,
                 periodic_prompt: int = 7,
                 upper_codebook_mask: int = 3):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.mask_token = mask_token
        self.periodic_prompt = periodic_prompt
        self.upper_codebook_mask = upper_codebook_mask
        
    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask and apply to codes.
        
        Args:
            codes: Input tokens [batch, n_codebooks, sequence_length]
            
        Returns:
            mask: Binary mask [batch, n_codebooks, sequence_length]
            masked_codes: Codes with mask tokens applied
        """
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        # Initialize mask with all ones (everything masked)
        mask = torch.ones(batch_size, n_codebooks, seq_len, 
                         dtype=torch.long, device=device)
        
        # Create periodic unmask pattern
        # Generate all positions and then mask based on modulo
        positions = torch.arange(seq_len, device=device)
        periodic_unmask = (positions % self.periodic_prompt) == 0
        
        # Create codebook mask - which codebooks to apply periodic pattern to
        codebook_indices = torch.arange(n_codebooks, device=device)
        apply_periodic = codebook_indices < self.upper_codebook_mask
        
        # Expand masks to full dimensions
        periodic_unmask = periodic_unmask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        periodic_unmask = periodic_unmask.expand(batch_size, n_codebooks, -1)
        
        apply_periodic = apply_periodic.unsqueeze(0).unsqueeze(-1)  # [1, n_codebooks, 1]
        apply_periodic = apply_periodic.expand(batch_size, -1, seq_len)
        
        # Apply periodic pattern only to lower codebooks
        mask = torch.where(
            apply_periodic,
            (~periodic_unmask).long(),  # Apply periodic pattern
            torch.ones_like(mask)       # Keep masked
        )
        
        # Apply mask to codes
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes


class FlexibleONNXMaskGenerator(nn.Module):
    """
    More flexible ONNX mask generator that takes parameters as inputs.
    """
    
    def __init__(self,
                 n_codebooks: int = 14,
                 mask_token: int = 1024):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.mask_token = mask_token
        
    def forward(self, 
                codes: torch.Tensor,
                periodic_prompt: torch.Tensor,
                upper_codebook_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask with parameters as tensors.
        
        Args:
            codes: Input tokens [batch, n_codebooks, sequence_length]
            periodic_prompt: Period as scalar tensor
            upper_codebook_mask: Number of lower codebooks as scalar tensor
            
        Returns:
            mask: Binary mask
            masked_codes: Codes with mask applied
        """
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        # Initialize mask
        mask = torch.ones(batch_size, n_codebooks, seq_len, 
                         dtype=torch.long, device=device)
        
        # Create periodic pattern using modulo
        positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
        positions = positions.expand(batch_size, n_codebooks, -1)
        
        # Periodic unmask pattern
        periodic_unmask = (positions % periodic_prompt) == 0
        
        # Create codebook range mask
        codebook_indices = torch.arange(n_codebooks, device=device).unsqueeze(-1)
        codebook_mask = codebook_indices < upper_codebook_mask
        codebook_mask = codebook_mask.unsqueeze(0).expand(batch_size, -1, seq_len)
        
        # Combine masks: unmask only where both conditions are true
        unmask = periodic_unmask & codebook_mask
        mask = ~unmask
        
        # Apply mask to codes
        masked_codes = torch.where(
            mask,
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask.long(), masked_codes