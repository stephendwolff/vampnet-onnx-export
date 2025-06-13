"""
Codec embedding extraction and management for VampNet ONNX models.

This module provides utilities for extracting embeddings from VampNet codecs
and converting them to ONNX-compatible formats.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodecEmbeddingExtractor:
    """Extracts and manages codec embeddings for ONNX models."""
    
    def __init__(self, codec_model_path: Optional[str] = None):
        """
        Initialize the codec embedding extractor.
        
        Args:
            codec_model_path: Path to codec model checkpoint (optional)
        """
        self.codec_model_path = codec_model_path
        self.codec = None
        
    def load_codec(self, codec_model_path: Optional[str] = None) -> None:
        """Load the VampNet codec model."""
        if codec_model_path:
            self.codec_model_path = codec_model_path
            
        if self.codec_model_path is None:
            raise ValueError("No codec model path provided")
            
        # Import VampNet dynamically
        try:
            from lac import LAC
            self.codec = LAC.load(self.codec_model_path)
            logger.info(f"Loaded LAC codec from {self.codec_model_path}")
        except ImportError:
            logger.warning("LAC not available, trying alternative loading method")
            # Alternative: load from checkpoint directly
            checkpoint = torch.load(self.codec_model_path, map_location='cpu')
            # Extract codec from checkpoint structure
            
    def extract_quantizer_embeddings(self) -> torch.Tensor:
        """
        Extract quantizer embeddings from codec.
        
        Returns:
            Tensor of shape [n_codebooks, vocab_size, latent_dim]
        """
        if self.codec is None:
            self.load_codec()
            
        embeddings = []
        
        # Extract from each quantizer
        for i, quantizer in enumerate(self.codec.quantizer.quantizers):
            if hasattr(quantizer, '_codebook'):
                codebook = quantizer._codebook.embed
            elif hasattr(quantizer, 'codebook'):
                codebook = quantizer.codebook.weight
            else:
                logger.warning(f"Could not find codebook for quantizer {i}")
                continue
                
            logger.info(f"Extracted codebook {i} with shape {codebook.shape}")
            embeddings.append(codebook)
            
        return torch.stack(embeddings) if embeddings else torch.empty(0)
        
    def create_onnx_embeddings(
        self,
        latent_dim: int = 128,
        model_dim: int = 1280,
        n_codebooks: int = 14
    ) -> Dict[str, torch.Tensor]:
        """
        Create ONNX-compatible embeddings from codec.
        
        Args:
            latent_dim: Dimension of codec latent space
            model_dim: Dimension of transformer model
            n_codebooks: Number of codebooks
            
        Returns:
            Dictionary containing embedding tensors
        """
        # Extract raw codebooks
        codebooks = self.extract_quantizer_embeddings()
        
        if codebooks.shape[0] == 0:
            logger.warning("No codebooks extracted, creating random embeddings")
            vocab_size = 1024
            codebooks = torch.randn(n_codebooks, vocab_size, latent_dim)
            
        vocab_size = codebooks.shape[1]
        
        # Create projection from latent to model dimension
        projection = torch.randn(latent_dim * n_codebooks, model_dim)
        projection = projection / (latent_dim * n_codebooks) ** 0.5
        
        # Create mask tokens (special token for each codebook)
        mask_tokens = torch.randn(n_codebooks, latent_dim)
        
        return {
            'codebooks': codebooks,
            'projection': projection,
            'mask_tokens': mask_tokens,
            'vocab_size': vocab_size,
            'latent_dim': latent_dim,
            'model_dim': model_dim
        }
        
    def create_embedding_layer(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        include_special_tokens: bool = True
    ) -> torch.Tensor:
        """
        Create a combined embedding layer for ONNX export.
        
        Args:
            embeddings_dict: Dictionary from create_onnx_embeddings
            include_special_tokens: Whether to include mask tokens
            
        Returns:
            Combined embedding tensor
        """
        codebooks = embeddings_dict['codebooks']
        mask_tokens = embeddings_dict['mask_tokens']
        vocab_size = embeddings_dict['vocab_size']
        
        n_codebooks, _, latent_dim = codebooks.shape
        
        if include_special_tokens:
            # Add mask token as the last token in vocabulary
            extended_codebooks = []
            for i in range(n_codebooks):
                # Concatenate regular tokens and mask token
                extended = torch.cat([
                    codebooks[i],
                    mask_tokens[i].unsqueeze(0)
                ], dim=0)
                extended_codebooks.append(extended)
            final_codebooks = torch.stack(extended_codebooks)
            final_vocab_size = vocab_size + 1
        else:
            final_codebooks = codebooks
            final_vocab_size = vocab_size
            
        logger.info(f"Created embedding layer with shape {final_codebooks.shape}")
        logger.info(f"Vocabulary size: {final_vocab_size} (includes special tokens: {include_special_tokens})")
        
        return final_codebooks


def extract_and_convert_embeddings(
    codec_path: str,
    model_dim: int = 1280,
    n_codebooks: int = 14,
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings from codec and convert to ONNX format.
    
    Args:
        codec_path: Path to codec checkpoint
        model_dim: Transformer model dimension
        n_codebooks: Number of codebooks to use
        save_path: Optional path to save embeddings
        
    Returns:
        Dictionary containing all embedding tensors
    """
    extractor = CodecEmbeddingExtractor(codec_path)
    embeddings = extractor.create_onnx_embeddings(
        model_dim=model_dim,
        n_codebooks=n_codebooks
    )
    
    # Create combined embedding layer
    embeddings['combined_embeddings'] = extractor.create_embedding_layer(
        embeddings,
        include_special_tokens=True
    )
    
    if save_path:
        torch.save(embeddings, save_path)
        logger.info(f"Saved embeddings to {save_path}")
        
    return embeddings


def load_embeddings_into_model(
    model: torch.nn.Module,
    embeddings: Dict[str, torch.Tensor],
    use_projection: bool = True
) -> None:
    """
    Load extracted embeddings into an ONNX model.
    
    Args:
        model: Target model with token_embedding layer
        embeddings: Dictionary from extract_and_convert_embeddings
        use_projection: Whether to apply projection matrix
    """
    combined = embeddings['combined_embeddings']
    n_codebooks = combined.shape[0]
    
    # Update model's embedding layer
    if hasattr(model, 'token_embedding'):
        model.token_embedding.weight.data = combined
        model.token_embedding.n_codebooks = n_codebooks
        model.token_embedding.vocab_size = combined.shape[1]
        logger.info(f"Loaded embeddings into model: {combined.shape}")
    else:
        logger.warning("Model does not have token_embedding layer")
        
    # Apply projection if needed
    if use_projection and 'projection' in embeddings:
        # This would typically be applied after embedding lookup
        # but before transformer layers
        logger.info("Projection matrix available for post-embedding transformation")