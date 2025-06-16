"""
Sampling utilities for VampNet ONNX models.
Provides temperature and top-p (nucleus) sampling for token generation.
"""

import numpy as np
from typing import Union, Optional


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to logits and return probabilities.
    
    Args:
        logits: Raw logits from the model [vocab_size]
        temperature: Temperature for scaling (higher = more random)
        
    Returns:
        Probabilities after temperature scaling
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = softmax(scaled_logits)
    
    return probs


def top_p_sample(probs: np.ndarray, top_p: float = 0.9) -> np.ndarray:
    """
    Apply top-p (nucleus) sampling to probabilities.
    
    Args:
        probs: Probability distribution [vocab_size]
        top_p: Cumulative probability threshold
        
    Returns:
        Filtered and renormalized probabilities
    """
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Calculate cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Find cutoff index
    cutoff_index = np.searchsorted(cumsum_probs, top_p, side='right')
    cutoff_index = max(0, cutoff_index)  # Ensure at least one token
    
    # Create filtered probability distribution
    filtered_probs = np.zeros_like(probs)
    filtered_probs[sorted_indices[:cutoff_index + 1]] = sorted_probs[:cutoff_index + 1]
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return filtered_probs


def sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None
) -> int:
    """
    Sample a token from logits using temperature and top-p/top-k sampling.
    
    Args:
        logits: Raw logits from the model [vocab_size]
        temperature: Temperature for scaling
        top_p: Top-p threshold for nucleus sampling
        top_k: Optional top-k filtering
        
    Returns:
        Sampled token index
    """
    # Apply temperature
    probs = temperature_sample(logits, temperature)
    
    # Apply top-k if specified
    if top_k is not None and top_k > 0:
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        filtered_probs = np.zeros_like(probs)
        filtered_probs[top_k_indices] = probs[top_k_indices]
        probs = filtered_probs / filtered_probs.sum()
    
    # Apply top-p
    probs = top_p_sample(probs, top_p)
    
    # Sample
    token = np.random.choice(len(probs), p=probs)
    
    return token


def batch_sample_from_logits(
    logits: np.ndarray,
    mask: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None
) -> np.ndarray:
    """
    Sample tokens from a batch of logits with masking.
    
    Args:
        logits: Logits from model [batch, n_codebooks, seq_len, vocab_size]
        mask: Boolean mask indicating positions to sample [batch, n_codebooks, seq_len]
        temperature: Temperature for scaling
        top_p: Top-p threshold
        top_k: Optional top-k filtering
        
    Returns:
        Sampled tokens [batch, n_codebooks, seq_len]
    """
    batch_size, n_codebooks, seq_len, vocab_size = logits.shape
    tokens = np.zeros((batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    for b in range(batch_size):
        for c in range(n_codebooks):
            for t in range(seq_len):
                if mask[b, c, t]:
                    # Sample new token
                    tokens[b, c, t] = sample_from_logits(
                        logits[b, c, t],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
    
    return tokens


def iterative_sample(
    session,
    codes: np.ndarray,
    mask: np.ndarray,
    n_steps: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    temperature_decay: float = 1.0
) -> np.ndarray:
    """
    Iteratively sample tokens using a model.
    
    Args:
        session: ONNX Runtime session
        codes: Initial codes [batch, n_codebooks, seq_len]
        mask: Positions to generate [batch, n_codebooks, seq_len]
        n_steps: Number of iterative refinement steps
        temperature: Initial temperature
        top_p: Top-p threshold
        top_k: Optional top-k filtering
        temperature_decay: Factor to decay temperature each step
        
    Returns:
        Generated codes
    """
    codes = codes.copy()
    current_temp = temperature
    
    for step in range(n_steps):
        # Get logits
        logits = session.run(None, {
            'codes': codes.astype(np.int64),
            'mask': mask
        })[0]
        
        # Sample
        sampled = batch_sample_from_logits(
            logits, mask, 
            temperature=current_temp,
            top_p=top_p,
            top_k=top_k
        )
        
        # Update codes
        codes[mask] = sampled[mask]
        
        # Decay temperature
        current_temp *= temperature_decay
    
    return codes