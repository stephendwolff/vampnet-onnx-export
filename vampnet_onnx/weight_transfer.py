"""
Weight transfer utilities for VampNet to ONNX model conversion.

This module provides functionality to transfer weights from pretrained VampNet models
to ONNX-compatible architectures, including codec embeddings, transformer weights,
and output projections.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightTransferManager:
    """Manages the complete weight transfer process from VampNet to ONNX models."""
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize the weight transfer manager.
        
        Args:
            checkpoint_path: Path to the VampNet checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.state_dict = None
        
    def load_checkpoint(self) -> None:
        """Load the VampNet checkpoint."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.state_dict = self.checkpoint.get('state_dict', self.checkpoint)
        
    def extract_codec_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Extract codec embeddings from the checkpoint.
        
        Returns:
            Dictionary containing:
            - 'codebooks': Tensor of shape [14, vocab_size, latent_dim]
            - 'mask_token': Tensor of shape [14, latent_dim]
            - 'embedding_proj': Optional projection matrix
        """
        if self.state_dict is None:
            self.load_checkpoint()
            
        embeddings = {}
        
        # Extract quantizer codebooks
        codebooks = []
        for i in range(14):  # VampNet uses 14 codebooks
            key = f'codec.quantizer.quantizers.{i}._codebook.embed'
            if key in self.state_dict:
                codebook = self.state_dict[key]
                logger.info(f"Extracted codebook {i} with shape {codebook.shape}")
                codebooks.append(codebook)
            else:
                logger.warning(f"Codebook {i} not found in checkpoint")
                
        if codebooks:
            embeddings['codebooks'] = torch.stack(codebooks)
            
        # Extract mask token embeddings
        mask_tokens = []
        for i in range(14):
            key = f'embedding.mask_token.{i}'
            if key in self.state_dict:
                mask_token = self.state_dict[key]
                logger.info(f"Extracted mask token {i} with shape {mask_token.shape}")
                mask_tokens.append(mask_token)
                
        if mask_tokens:
            embeddings['mask_token'] = torch.stack(mask_tokens)
            
        # Extract embedding projection if exists
        if 'embedding.proj' in self.state_dict:
            embeddings['embedding_proj'] = self.state_dict['embedding.proj']
            logger.info(f"Extracted embedding projection with shape {embeddings['embedding_proj'].shape}")
            
        return embeddings
    
    def transfer_transformer_weights(self, model: torch.nn.Module, is_c2f: bool = False) -> None:
        """
        Transfer transformer weights from checkpoint to ONNX model.
        
        Args:
            model: Target ONNX-compatible transformer model
            is_c2f: Whether this is a coarse-to-fine model
        """
        if self.state_dict is None:
            self.load_checkpoint()
            
        prefix = 'c2f.' if is_c2f else ''
        
        # Transfer transformer layers
        n_layers = model.n_layers
        for i in range(n_layers):
            # Layer normalization
            self._transfer_norm_weights(
                f'{prefix}transformer.layers.{i}.norm1',
                model.transformer[i].norm1
            )
            self._transfer_norm_weights(
                f'{prefix}transformer.layers.{i}.norm2',
                model.transformer[i].norm2
            )
            
            # Self-attention
            self._transfer_attention_weights(
                f'{prefix}transformer.layers.{i}.self_attn',
                model.transformer[i].self_attn
            )
            
            # Feed-forward network
            self._transfer_ffn_weights(
                f'{prefix}transformer.layers.{i}.ffn',
                model.transformer[i].ffn
            )
            
        # Final layer norm
        self._transfer_norm_weights(
            f'{prefix}transformer.final_norm',
            model.final_norm
        )
        
        logger.info(f"Transferred weights for {n_layers} transformer layers")
        
    def transfer_output_projections(self, model: torch.nn.Module, is_c2f: bool = False) -> None:
        """
        Transfer output projection weights.
        
        Args:
            model: Target ONNX-compatible model
            is_c2f: Whether this is a coarse-to-fine model
        """
        if self.state_dict is None:
            self.load_checkpoint()
            
        prefix = 'c2f.' if is_c2f else ''
        
        # Transfer output projections for each codebook
        n_codebooks = 10 if is_c2f else 4
        for i in range(n_codebooks):
            proj_key = f'{prefix}classifier.{i}.weight'
            if proj_key in self.state_dict:
                weight = self.state_dict[proj_key]
                # Handle dimension mismatch if necessary
                if weight.shape[0] == 4096 and model.output_proj[i].out_features == 1025:
                    weight = weight[:1025]
                model.output_proj[i].weight.data = weight
                logger.info(f"Transferred output projection {i} with shape {weight.shape}")
                
    def _transfer_norm_weights(self, source_key: str, target_layer: torch.nn.Module) -> None:
        """Transfer normalization layer weights."""
        weight_key = f'{source_key}.weight'
        if weight_key in self.state_dict:
            target_layer.weight.data = self.state_dict[weight_key]
            
    def _transfer_attention_weights(self, source_key: str, target_layer: torch.nn.Module) -> None:
        """Transfer multi-head attention weights."""
        # Combined QKV projection
        qkv_key = f'{source_key}.qkv_proj.weight'
        if qkv_key in self.state_dict:
            target_layer.qkv_proj.weight.data = self.state_dict[qkv_key]
            
        # Output projection
        out_key = f'{source_key}.o_proj.weight'
        if out_key in self.state_dict:
            target_layer.o_proj.weight.data = self.state_dict[out_key]
            
    def _transfer_ffn_weights(self, source_key: str, target_layer: torch.nn.Module) -> None:
        """Transfer feed-forward network weights."""
        # Gate and up projections (for GatedGELU)
        gate_key = f'{source_key}.w_gated.weight'
        if gate_key in self.state_dict:
            target_layer.w_gated.weight.data = self.state_dict[gate_key]
            
        up_key = f'{source_key}.w_up.weight'
        if up_key in self.state_dict:
            target_layer.w_up.weight.data = self.state_dict[up_key]
            
        # Down projection
        down_key = f'{source_key}.w_down.weight'
        if down_key in self.state_dict:
            target_layer.w_down.weight.data = self.state_dict[down_key]


def complete_weight_transfer(
    checkpoint_path: str,
    coarse_model: Optional[torch.nn.Module] = None,
    c2f_model: Optional[torch.nn.Module] = None,
    return_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Perform complete weight transfer from VampNet checkpoint.
    
    Args:
        checkpoint_path: Path to VampNet checkpoint
        coarse_model: Coarse transformer model (optional)
        c2f_model: Coarse-to-fine transformer model (optional)
        return_embeddings: Whether to return extracted embeddings
        
    Returns:
        Dictionary containing transferred models and embeddings
    """
    manager = WeightTransferManager(checkpoint_path)
    manager.load_checkpoint()
    
    results = {}
    
    # Extract codec embeddings
    if return_embeddings:
        results['embeddings'] = manager.extract_codec_embeddings()
    
    # Transfer coarse model weights
    if coarse_model is not None:
        manager.transfer_transformer_weights(coarse_model, is_c2f=False)
        manager.transfer_output_projections(coarse_model, is_c2f=False)
        results['coarse_model'] = coarse_model
        logger.info("Completed weight transfer for coarse model")
    
    # Transfer C2F model weights
    if c2f_model is not None:
        manager.transfer_transformer_weights(c2f_model, is_c2f=True)
        manager.transfer_output_projections(c2f_model, is_c2f=True)
        results['c2f_model'] = c2f_model
        logger.info("Completed weight transfer for C2F model")
    
    return results