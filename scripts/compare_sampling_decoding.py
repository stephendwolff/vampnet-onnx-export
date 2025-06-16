#!/usr/bin/env python3
"""
Step 6: Compare Sampling/Decoding - logits to tokens conversion.
Compare how VampNet and ONNX convert transformer outputs to tokens.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm
from scripts.export_vampnet_transformer_v11_fixed import VampNetTransformerV11, transfer_weights_v11

print("="*80)
print("STEP 6: SAMPLING/DECODING COMPARISON")
print("="*80)

# Load models
print("\n1. Loading models...")
torch.manual_seed(42)
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
remove_weight_norm(vampnet.classifier.layers[0])
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Check VampNet's sampling methods
print("\n2. Checking VampNet's sampling methods...")
print(f"VampNet has sample method: {hasattr(vampnet, 'sample')}")
print(f"VampNet has generate method: {hasattr(vampnet, 'generate')}")

# Look for sampling-related methods
vampnet_methods = [m for m in dir(vampnet) if not m.startswith('_')]
sampling_methods = [m for m in vampnet_methods if 'sample' in m.lower() or 'generate' in m.lower() or 'decode' in m.lower()]
print(f"Sampling-related methods: {sampling_methods}")

# Check the sampling implementation
if hasattr(vampnet, 'sample'):
    import inspect
    print("\n3. VampNet.sample signature:")
    print(inspect.signature(vampnet.sample))
    
    # Get sample method source to understand parameters
    try:
        sample_source = inspect.getsource(vampnet.sample)
        # Look for key parameters
        if 'temperature' in sample_source:
            print("  - Uses temperature sampling")
        if 'top_k' in sample_source:
            print("  - Uses top-k sampling")
        if 'top_p' in sample_source:
            print("  - Uses top-p (nucleus) sampling")
        if 'softmax' in sample_source:
            print("  - Uses softmax")
        if 'multinomial' in sample_source:
            print("  - Uses multinomial sampling")
        if 'argmax' in sample_source:
            print("  - Uses argmax (greedy) sampling")
    except:
        pass

# Test basic sampling
print("\n4. Testing basic sampling...")

# Create test input
codes = torch.randint(0, 1024, (1, 4, 20))
mask = torch.zeros((1, 4, 20), dtype=torch.bool)
mask[:, :, 10:] = True  # Mask last half
masked_codes = codes.clone()
masked_codes[mask] = 1024

# Get transformer output
with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Try different sampling approaches
    print("\n5. Testing different sampling approaches...")
    
    # Get logits from our V11 model
    model = VampNetTransformerV11()
    transfer_weights_v11("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
    model.eval()
    
    logits = model(latents)  # [batch, codebooks, seq_len, vocab_size+1]
    print(f"Logits shape: {logits.shape}")
    
    # Remove mask token logits for sampling
    logits_no_mask = logits[:, :, :, :1024]  # [batch, codebooks, seq_len, vocab_size]
    
    # Test different sampling methods
    print("\n6. Sampling methods comparison:")
    
    # Method 1: Argmax (greedy)
    tokens_argmax = torch.argmax(logits_no_mask, dim=-1)
    print(f"  Argmax tokens shape: {tokens_argmax.shape}")
    print(f"  Sample tokens (first 5): {tokens_argmax[0, 0, :5]}")
    
    # Method 2: Softmax + multinomial (temperature = 1.0)
    probs = torch.softmax(logits_no_mask, dim=-1)
    tokens_multinomial = torch.multinomial(probs.reshape(-1, 1024), 1).reshape(1, 4, 20)
    print(f"  Multinomial tokens shape: {tokens_multinomial.shape}")
    print(f"  Sample tokens (first 5): {tokens_multinomial[0, 0, :5]}")
    
    # Method 3: Temperature sampling
    temperature = 0.8
    logits_temp = logits_no_mask / temperature
    probs_temp = torch.softmax(logits_temp, dim=-1)
    tokens_temp = torch.multinomial(probs_temp.reshape(-1, 1024), 1).reshape(1, 4, 20)
    print(f"  Temperature ({temperature}) tokens shape: {tokens_temp.shape}")
    print(f"  Sample tokens (first 5): {tokens_temp[0, 0, :5]}")
    
    # Method 4: Top-k sampling
    k = 50
    top_k_logits, top_k_indices = torch.topk(logits_no_mask, k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    # Sample from top-k
    sampled_indices = torch.multinomial(top_k_probs.reshape(-1, k), 1)
    # Convert back to token indices
    tokens_topk = torch.gather(top_k_indices.reshape(-1, k), 1, sampled_indices)
    tokens_topk = tokens_topk.reshape(1, 4, 20)
    print(f"  Top-k ({k}) tokens shape: {tokens_topk.shape}")
    print(f"  Sample tokens (first 5): {tokens_topk[0, 0, :5]}")

# Check if VampNet has specific sampling parameters
print("\n7. Checking VampNet's actual sampling behavior...")

# Try to use VampNet's sample method if it exists
if hasattr(vampnet, 'sample'):
    try:
        # Common parameters for sampling
        sample_params = {
            'codes': masked_codes,
            'mask': mask,
            'temperature': 1.0,
            'top_k': None,
            'top_p': None,
        }
        
        # Try calling sample with different parameter combinations
        try:
            vampnet_output = vampnet.sample(**sample_params)
            print("  VampNet sample succeeded with all parameters")
        except TypeError as e:
            print(f"  VampNet sample parameter error: {e}")
            # Try with fewer parameters
            try:
                vampnet_output = vampnet.sample(masked_codes, mask)
                print("  VampNet sample succeeded with just codes and mask")
            except:
                pass
                
    except Exception as e:
        print(f"  Error calling VampNet sample: {e}")

# Save sampling comparison results
print("\n8. Summary of sampling methods:")
print("  - Argmax: Deterministic, always picks highest probability token")
print("  - Multinomial: Stochastic, samples from full distribution")
print("  - Temperature: Controls randomness (lower = more deterministic)")
print("  - Top-k: Samples only from k most likely tokens")
print("  - Top-p: Samples from tokens that sum to probability p")

# Create a simple sampling function for ONNX
print("\n9. Creating ONNX-compatible sampling function...")

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample tokens from logits with various sampling strategies.
    
    Args:
        logits: [batch, codebooks, seq_len, vocab_size] 
        temperature: Sampling temperature
        top_k: Sample only from top k tokens
        top_p: Sample from tokens that sum to probability p
        
    Returns:
        tokens: [batch, codebooks, seq_len]
    """
    batch, codebooks, seq_len, vocab_size = logits.shape
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab_size)
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
    
    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
    
    # Sample
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.reshape(-1, vocab_size), 1)
    tokens = tokens.reshape(batch, codebooks, seq_len)
    
    return tokens

# Test the sampling function
test_logits = logits[:, :, :, :1024]
sampled_tokens = sample_from_logits(test_logits, temperature=0.8, top_k=50)
print(f"Sampled tokens shape: {sampled_tokens.shape}")
print(f"Sample tokens: {sampled_tokens[0, 0, :5]}")

print("\n✅ Sampling/Decoding comparison complete!")
print("   Key finding: Multiple sampling strategies available")
print("   Next step: Codec Decoder comparison (Step 7)")