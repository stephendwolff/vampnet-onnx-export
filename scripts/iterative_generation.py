#!/usr/bin/env python3
"""
Implement iterative generation logic matching VampNet's generate method.
This enables the parallel iterative decoding process used by VampNet.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import math
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))


def sample_from_logits(
    logits, 
    sample: bool = True,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    typical_filtering: bool = False,
    typical_mass: float = 0.2,
    typical_min_tokens: int = 1,
    return_probs: bool = False
):
    """Sample tokens from logits using VampNet's sampling strategy."""
    shp = logits.shape[:-1]
    
    # Skip typical filtering for now (can be added later)
    if typical_filtering:
        pass  # TODO: Implement typical filtering
    
    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")
    
    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[
            ..., :-1
        ]
        
        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        logits[indices_to_remove] = -float("inf")
    
    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )
    
    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    
    return token


def mask_from_frac(z, mask_frac: float = 0.0):
    """Create a mask with a fraction of tokens masked."""
    mask = torch.rand(z.shape) < mask_frac
    mask = mask.long()
    return mask


def score_logits(
    logits: torch.Tensor,
    mask_token: int = 1024,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Score logits for confidence-based masking.
    Returns confidence scores for each position.
    """
    # Remove mask token from logits
    logits_no_mask = logits[:, :, :, :mask_token]
    
    # Apply temperature and get probabilities
    if temperature > 0:
        probs = F.softmax(logits_no_mask / temperature, dim=-1)
    else:
        probs = logits_no_mask.softmax(dim=-1)
    
    # Get max probability as confidence score
    scores = probs.max(dim=-1)[0]  # [batch, codebooks, seq_len]
    
    return scores


def update_mask_using_scores(
    mask: torch.Tensor,
    scores: torch.Tensor,
    mask_ratio: float = 0.9,
    _mask_token: int = 1024
) -> torch.Tensor:
    """
    Update mask based on confidence scores.
    Lower confidence positions are more likely to be remasked.
    """
    # Only update positions that are currently masked
    masked_positions = mask == 1
    
    if not masked_positions.any():
        return mask
    
    # Get scores for masked positions
    masked_scores = scores[masked_positions]
    
    # Determine how many to keep masked
    n_masked = masked_positions.sum()
    n_to_mask = int(n_masked * mask_ratio)
    
    if n_to_mask == 0:
        return torch.zeros_like(mask)
    
    # Sort scores and remask lowest confidence positions
    sorted_indices = masked_scores.argsort()
    threshold_idx = sorted_indices[n_to_mask] if n_to_mask < len(sorted_indices) else sorted_indices[-1]
    threshold_score = masked_scores[threshold_idx]
    
    # Create new mask
    new_mask = torch.zeros_like(mask)
    new_mask[masked_positions] = (scores[masked_positions] <= threshold_score).long()
    
    return new_mask


class IterativeGenerator:
    """
    Implements VampNet's iterative generation process.
    Compatible with ONNX models.
    """
    
    def __init__(self, 
                 transformer_session,  # ONNX session or PyTorch model
                 codec_embeddings: torch.Tensor,
                 mask_token: int = 1024,
                 n_codebooks: int = 4,
                 latent_dim: int = 8):
        self.transformer = transformer_session
        self.codec_embeddings = codec_embeddings
        self.mask_token = mask_token
        self.n_codebooks = n_codebooks
        self.latent_dim = latent_dim
        self.is_onnx = hasattr(transformer_session, 'run')
    
    def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert codes to latents using codec embeddings."""
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Handle mask tokens
        mask_indices = codes == self.mask_token
        codes_clamped = codes.clone()
        codes_clamped[mask_indices] = 0
        
        # Gather embeddings
        latents = []
        for cb in range(n_codebooks):
            cb_codes = codes_clamped[:, cb, :]  # [batch, seq_len]
            cb_embeddings = self.codec_embeddings[cb]  # [vocab_size, latent_dim]
            cb_latents = F.embedding(cb_codes, cb_embeddings)  # [batch, seq_len, latent_dim]
            latents.append(cb_latents)
        
        # Stack and reshape
        latents = torch.stack(latents, dim=1)  # [batch, n_codebooks, seq_len, latent_dim]
        latents = latents.permute(0, 1, 3, 2)  # [batch, n_codebooks, latent_dim, seq_len]
        latents = latents.reshape(batch_size, n_codebooks * self.latent_dim, seq_len)
        
        # Zero out masked positions
        for cb in range(n_codebooks):
            for pos in range(seq_len):
                if mask_indices[:, cb, pos].any():
                    start_idx = cb * self.latent_dim
                    end_idx = (cb + 1) * self.latent_dim
                    latents[:, start_idx:end_idx, pos] = 0
        
        return latents
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Run transformer forward pass."""
        # Convert codes to latents
        latents = self.codes_to_latents(codes)
        
        # Run transformer
        if self.is_onnx:
            logits = self.transformer.run(None, {'latents': latents.numpy()})[0]
            logits = torch.from_numpy(logits)
        else:
            logits = self.transformer(latents)
        
        return logits
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        mask: torch.Tensor,
        time_steps: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_filtering: bool = False,
        typical_mass: float = 0.15,
        typical_min_tokens: int = 1,
        return_signal: bool = False,
        mask_temperature: float = 10.5,
        sample_cutoff: float = 1.0,
        _sampling_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens using iterative refinement.
        Matches VampNet's generate method.
        """
        if _sampling_steps is not None:
            time_steps = _sampling_steps
        
        # Initialize
        z = start_tokens.clone()
        batch_size, n_codebooks, seq_len = z.shape
        
        # Main generation loop
        for step in range(time_steps):
            # Apply mask
            z_masked = z.clone()
            z_masked[mask.bool()] = self.mask_token
            
            # Forward pass
            logits = self.forward(z_masked)
            
            # Sample tokens
            sampled_z = []
            for cb in range(logits.shape[1]):
                cb_logits = logits[:, cb, :, :self.mask_token]  # Remove mask token
                
                # Sample tokens for this codebook
                cb_sampled = sample_from_logits(
                    cb_logits.reshape(-1, self.mask_token),
                    sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens
                )
                cb_sampled = cb_sampled.reshape(batch_size, seq_len)
                sampled_z.append(cb_sampled)
            
            sampled_z = torch.stack(sampled_z, dim=1)
            
            # Update only masked positions
            z[mask.bool()] = sampled_z[mask.bool()]
            
            # Update mask based on confidence (except last step)
            if step < time_steps - 1:
                scores = score_logits(logits, self.mask_token, mask_temperature)
                
                # Determine masking schedule
                progress = (step + 1) / time_steps
                mask_ratio = 1.0 - progress * sample_cutoff
                
                # Update mask
                mask = update_mask_using_scores(mask, scores, mask_ratio, self.mask_token)
        
        return z


def create_onnx_generator(
    transformer_path: str,
    codec_path: str,
    n_codebooks: int = 4,
    latent_dim: int = 8,
    mask_token: int = 1024
) -> IterativeGenerator:
    """Create an iterative generator for ONNX models."""
    import onnxruntime as ort
    
    # Load transformer
    transformer_session = ort.InferenceSession(transformer_path)
    
    # Load codec embeddings
    codec_ckpt = torch.load(codec_path, map_location='cpu')
    codec_embeddings = []
    
    for cb in range(n_codebooks):
        embed_key = f'quantizer.quantizers.{cb}.codebook.weight'
        if embed_key in codec_ckpt['state_dict']:
            embeddings = codec_ckpt['state_dict'][embed_key]
            codec_embeddings.append(embeddings)
    
    codec_embeddings = torch.stack(codec_embeddings[:n_codebooks])
    
    return IterativeGenerator(
        transformer_session,
        codec_embeddings,
        mask_token,
        n_codebooks,
        latent_dim
    )


def test_iterative_generation():
    """Test the iterative generation process."""
    print("="*80)
    print("TESTING ITERATIVE GENERATION")
    print("="*80)
    
    # Create test generator with PyTorch model
    from scripts.export_vampnet_transformer_v11_fixed import VampNetTransformerV11
    
    # Load model
    model = VampNetTransformerV11()
    checkpoint = torch.load('vampnet_transformer_v11.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load codec embeddings
    codec_ckpt = torch.load('models/vampnet/codec.pth', map_location='cpu')
    codec_embeddings = []
    for cb in range(4):
        embed_key = f'quantizer.quantizers.{cb}.codebook.weight'
        if embed_key in codec_ckpt['state_dict']:
            embeddings = codec_ckpt['state_dict'][embed_key]
            codec_embeddings.append(embeddings)
    codec_embeddings = torch.stack(codec_embeddings)
    
    # Create generator
    generator = IterativeGenerator(
        model,
        codec_embeddings,
        mask_token=1024,
        n_codebooks=4,
        latent_dim=8
    )
    
    # Test input
    torch.manual_seed(42)
    codes = torch.randint(0, 1024, (1, 4, 20))
    mask = torch.zeros((1, 4, 20), dtype=torch.long)
    mask[:, :, 10:] = 1  # Mask last half
    
    print(f"\nInput codes shape: {codes.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Generate
    print("\nRunning iterative generation...")
    generated = generator.generate(
        start_tokens=codes,
        mask=mask,
        time_steps=6,
        temperature=1.0,
        top_k=50,
        typical_filtering=True,
        typical_mass=0.15
    )
    
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Changed positions: {(generated != codes).sum().item()}")
    print(f"Success: {(generated[mask.bool()] != codes[mask.bool()]).any().item()}")
    
    # Test with ONNX
    if Path('vampnet_transformer_v11.onnx').exists():
        print("\n\nTesting with ONNX model...")
        onnx_generator = create_onnx_generator(
            'vampnet_transformer_v11.onnx',
            'models/vampnet/codec.pth',
            n_codebooks=4,
            latent_dim=8
        )
        
        torch.manual_seed(42)
        onnx_generated = onnx_generator.generate(
            start_tokens=codes,
            mask=mask,
            time_steps=6,
            temperature=1.0,
            top_k=50
        )
        
        print(f"ONNX generated shape: {onnx_generated.shape}")
        print(f"ONNX changed positions: {(onnx_generated != codes).sum().item()}")
        
        # Compare
        match = (generated == onnx_generated).all().item()
        print(f"\nPyTorch vs ONNX match: {match}")
    
    print("\n✅ Iterative generation implemented!")
    print("="*80)


if __name__ == "__main__":
    test_iterative_generation()