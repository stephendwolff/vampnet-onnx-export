"""
Mask generation module for VampNet ONNX export.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MaskGenerator(nn.Module):
    """
    Generates masks for VampNet token generation.
    ONNX-compatible version with deterministic operations.
    """
    
    def __init__(self,
                 n_codebooks: int = 14,
                 mask_token: int = 1024):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.mask_token = mask_token
        
    def forward(self,
                codes: torch.Tensor,
                periodic_prompt: int,
                upper_codebook_mask: int,
                offset: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask and apply to codes.
        
        Args:
            codes: Input tokens [batch, n_codebooks, sequence_length]
            periodic_prompt: Keep every Nth token (must be > 0)
            upper_codebook_mask: Number of lower codebooks to apply masking to
            offset: Optional offset for periodic pattern (for deterministic behavior)
            
        Returns:
            mask: Binary mask [batch, n_codebooks, sequence_length]
            masked_codes: Codes with mask tokens applied
        """
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        # Initialize mask with all ones (everything masked)
        mask = torch.ones(batch_size, n_codebooks, seq_len, 
                         dtype=torch.long, device=device)
        
        # Apply periodic pattern
        if periodic_prompt > 0:
            # Use provided offset or default to 0 for ONNX compatibility
            start_offset = offset if offset is not None else 0
            
            # Create indices for unmasked positions
            indices = torch.arange(start_offset, seq_len, periodic_prompt, device=device)
            
            # Ensure indices are within bounds
            valid_mask = indices < seq_len
            indices = indices[valid_mask]
            
            # Set periodic positions to 0 (unmasked)
            # Use masked_scatter to avoid conditional
            if indices.shape[0] > 0:  # This check is on shape, not tensor value
                mask[:, :, indices] = 0
        
        # Apply codebook masking
        # Always apply masking, but it will have no effect if upper_codebook_mask is 0
        if upper_codebook_mask > 0:
            # Mask all upper codebooks completely
            mask[:, upper_codebook_mask:, :] = 1
            
        # Apply mask to codes
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes
    
    @torch.jit.export
    def create_periodic_mask(self,
                           batch_size: int,
                           seq_len: int,
                           periodic_prompt: int,
                           offset: int = 0) -> torch.Tensor:
        """
        Create just the periodic mask pattern.
        Useful for visualization and debugging.
        """
        mask = torch.ones(batch_size, self.n_codebooks, seq_len, dtype=torch.long)
        
        if periodic_prompt > 0:
            for i in range(offset, seq_len, periodic_prompt):
                mask[:, :, i] = 0
                
        return mask


class AdvancedMaskGenerator(nn.Module):
    """
    Advanced mask generator with multiple masking strategies.
    """
    
    def __init__(self,
                 n_codebooks: int = 14,
                 mask_token: int = 1024):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.mask_token = mask_token
        
    def forward(self,
                codes: torch.Tensor,
                mask_type: str = "periodic",
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask with specified strategy.
        
        Args:
            codes: Input tokens [batch, n_codebooks, sequence_length]
            mask_type: Type of mask ("periodic", "random", "block", "hybrid")
            **kwargs: Additional arguments for specific mask types
            
        Returns:
            mask: Binary mask
            masked_codes: Codes with mask applied
        """
        if mask_type == "periodic":
            return self.periodic_mask(codes, **kwargs)
        elif mask_type == "random":
            return self.random_mask(codes, **kwargs)
        elif mask_type == "block":
            return self.block_mask(codes, **kwargs)
        elif mask_type == "hybrid":
            return self.hybrid_mask(codes, **kwargs)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
            
    def periodic_mask(self,
                     codes: torch.Tensor,
                     period: int = 7,
                     offset: int = 0,
                     upper_codebook_mask: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Periodic masking pattern."""
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        mask = torch.ones_like(codes, dtype=torch.long)
        
        # Apply periodic pattern
        for i in range(offset, seq_len, period):
            mask[:, :, i] = 0
            
        # Apply codebook masking
        if upper_codebook_mask > 0:
            mask[:, upper_codebook_mask:, :] = 1
            
        # Apply mask
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes
    
    def random_mask(self,
                   codes: torch.Tensor,
                   mask_ratio: float = 0.7,
                   upper_codebook_mask: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random masking pattern (deterministic for ONNX)."""
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        # For ONNX compatibility, use a deterministic pattern
        # In practice, you'd generate this outside and pass it in
        mask = torch.ones_like(codes, dtype=torch.long)
        
        # Simple deterministic "random" pattern
        n_keep = int(seq_len * (1 - mask_ratio))
        stride = max(1, seq_len // n_keep)
        
        for i in range(0, seq_len, stride):
            mask[:, :upper_codebook_mask, i] = 0
            
        # Apply codebook masking
        if upper_codebook_mask > 0:
            mask[:, upper_codebook_mask:, :] = 1
            
        # Apply mask
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes
    
    def block_mask(self,
                  codes: torch.Tensor,
                  block_size: int = 10,
                  n_blocks: int = 5,
                  upper_codebook_mask: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block masking pattern."""
        batch_size, n_codebooks, seq_len = codes.shape
        device = codes.device
        
        mask = torch.zeros_like(codes, dtype=torch.long)
        
        # Create block pattern
        block_stride = max(1, seq_len // n_blocks)
        
        for i in range(n_blocks):
            start = i * block_stride
            end = min(start + block_size, seq_len)
            mask[:, :upper_codebook_mask, start:end] = 1
            
        # Apply codebook masking
        if upper_codebook_mask > 0:
            mask[:, upper_codebook_mask:, :] = 1
            
        # Apply mask
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes
    
    def hybrid_mask(self,
                   codes: torch.Tensor,
                   periodic_period: int = 7,
                   block_size: int = 5,
                   upper_codebook_mask: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hybrid masking combining periodic and block patterns."""
        # Get periodic mask
        periodic_mask, _ = self.periodic_mask(
            codes, period=periodic_period, 
            upper_codebook_mask=upper_codebook_mask
        )
        
        # Get block mask
        block_mask, _ = self.block_mask(
            codes, block_size=block_size,
            upper_codebook_mask=upper_codebook_mask
        )
        
        # Combine masks (OR operation - mask if either masks)
        mask = torch.maximum(periodic_mask, block_mask)
        
        # Apply mask
        masked_codes = torch.where(
            mask.bool(),
            torch.full_like(codes, self.mask_token),
            codes
        )
        
        return mask, masked_codes