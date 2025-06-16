#!/usr/bin/env python3
"""
Pipeline Comparison: Step 5
Compare Transformer Forward Pass - token prediction
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("STEP 5: TRANSFORMER FORWARD PASS COMPARISON")
print("=" * 80)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Import interfaces
from vampnet_onnx import Interface as ONNXInterface
from vampnet_onnx.interface import AudioSignalCompat
from vampnet.interface import Interface as VampNetInterface

try:
    from audiotools import AudioSignal
    AUDIOTOOLS_AVAILABLE = True
except ImportError:
    AUDIOTOOLS_AVAILABLE = False

# Initialize interfaces
print("\nInitializing interfaces...")
device = torch.device('cpu')

# ONNX interface
onnx_interface = ONNXInterface.from_default_models(device='cpu')
print("✓ ONNX interface initialized")

# VampNet interface
vampnet_interface = VampNetInterface(
    codec_ckpt="../models/vampnet/codec.pth",
    coarse_ckpt="../models/vampnet/coarse.pth",
    coarse2fine_ckpt="../models/vampnet/c2f.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth",
)
vampnet_interface.to(device)
print("✓ VampNet interface initialized")

# Create test setup
print("\nCreating test data...")

# Create test audio for encoding
sample_rate = 44100
duration = 1.74  # ~100 tokens
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Create audio signal
if AUDIOTOOLS_AVAILABLE:
    audio_signal = AudioSignal(audio[np.newaxis, :], sample_rate)
else:
    audio_signal = AudioSignalCompat(audio, sample_rate)

# Encode to get real tokens
print("\nEncoding audio to tokens...")
tokens = onnx_interface.encode(audio_signal)
print(f"Encoded tokens shape: {tokens.shape}")

# Create a mask (mask middle section)
mask = np.zeros_like(tokens, dtype=bool)
mask_start = 30
mask_end = 70
mask[:, :, mask_start:mask_end] = True
masked_positions = np.sum(mask)
print(f"Mask shape: {mask.shape}")
print(f"Masked positions: {masked_positions} ({masked_positions / mask.size * 100:.1f}%)")

# Prepare masked tokens (replace with mask token)
MASK_TOKEN = 1024  # VampNet's mask token
masked_tokens = tokens.copy()
masked_tokens[mask] = MASK_TOKEN

print("\n" + "-" * 60)
print("TRANSFORMER FORWARD PASS COMPARISON")
print("-" * 60)

# Test both coarse and coarse-to-fine models
model_configs = [
    {
        "name": "Coarse Model",
        "onnx_model": onnx_interface.coarse_session,
        "vampnet_fn": vampnet_interface.coarse_vamp,
        "n_codebooks": 4,  # Coarse uses only first 4 codebooks
    },
    {
        "name": "Coarse-to-Fine Model",
        "onnx_model": onnx_interface.c2f_session,
        "vampnet_fn": vampnet_interface.coarse_to_fine,
        "n_codebooks": 14,  # C2F uses all codebooks
        "n_conditioning_codebooks": 4,  # First 4 are conditioning
    }
]

comparison_results = []

for config in model_configs:
    print(f"\n--- {config['name']} ---")
    
    # Prepare inputs for this model
    model_tokens = masked_tokens[:, :config['n_codebooks'], :].copy()
    model_mask = mask[:, :config['n_codebooks'], :].copy()
    
    # For C2F, we need to have coarse tokens already generated
    if config['name'] == "Coarse-to-Fine Model":
        # Use original tokens for first 4 codebooks (simulating coarse generation)
        model_tokens[:, :4, :] = tokens[:, :4, :]
        # Only mask codebooks 4-13
        model_mask[:, :4, :] = False
    
    print(f"Model input shape: {model_tokens.shape}")
    print(f"Masked positions for this model: {np.sum(model_mask)}")
    
    # Run ONNX model
    print("\nRunning ONNX model...")
    start_time = time.time()
    
    if config['onnx_model'] is not None:
        onnx_logits = config['onnx_model'].run(None, {
            'codes': model_tokens.astype(np.int64),
            'mask': model_mask
        })[0]
        onnx_time = time.time() - start_time
        print(f"ONNX output shape: {onnx_logits.shape}")
        print(f"ONNX inference time: {onnx_time:.3f}s")
    else:
        print("ONNX model not available")
        onnx_logits = None
        onnx_time = None
    
    # Run VampNet model
    print("\nRunning VampNet model...")
    torch_tokens = torch.tensor(model_tokens, dtype=torch.long, device=device)
    torch_mask = torch.tensor(model_mask, dtype=torch.bool, device=device)
    
    start_time = time.time()
    
    # Get logits from VampNet using the generate method
    with torch.no_grad():
        if config['name'] == "Coarse Model":
            # Use VampNet's generate method to get logits
            vampnet_model = vampnet_interface.coarse
            # VampNet's generate method handles embeddings internally
            # We need to convert tokens to embeddings first
            z_masked = torch_tokens
            # Embed the tokens
            latents = vampnet_model.embedding.from_codes(z_masked, vampnet_interface.codec)
            # Get logits
            logits = vampnet_model(latents)
            vampnet_logits = logits.cpu().numpy()
        else:
            # For C2F model
            vampnet_model = vampnet_interface.c2f
            z_masked = torch_tokens
            # Embed the tokens - C2F expects all 14 codebooks
            latents = vampnet_model.embedding.from_codes(z_masked, vampnet_interface.codec)
            # Get logits
            logits = vampnet_model(latents)
            vampnet_logits = logits.cpu().numpy()
    
    vampnet_time = time.time() - start_time
    print(f"VampNet output shape: {vampnet_logits.shape}")
    print(f"VampNet inference time: {vampnet_time:.3f}s")
    
    # Compare outputs
    if onnx_logits is not None:
        print("\n" + "-" * 40)
        print("COMPARISON RESULTS")
        print("-" * 40)
        
        # Shape comparison
        shape_match = onnx_logits.shape == vampnet_logits.shape
        print(f"Shape match: {shape_match}")
        if not shape_match:
            print(f"  ONNX shape: {onnx_logits.shape}")
            print(f"  VampNet shape: {vampnet_logits.shape}")
            # VampNet outputs logits in a different format
            # It outputs [batch, vocab_size, total_positions]
            # We need to reshape it to [batch, n_codebooks, seq_len, vocab_size]
            if len(vampnet_logits.shape) == 3 and vampnet_logits.shape[1] == 1024:
                # Reshape VampNet output
                batch_size = vampnet_logits.shape[0]
                vocab_size = vampnet_logits.shape[1]
                total_positions = vampnet_logits.shape[2]
                
                # For C2F model, it only outputs logits for fine codebooks
                if config['name'] == "Coarse-to-Fine Model":
                    n_fine_codebooks = config['n_codebooks'] - config['n_conditioning_codebooks']  # 10
                    seq_len = total_positions // n_fine_codebooks
                    
                    print(f"  C2F model: reshaping from {vampnet_logits.shape}")
                    print(f"  Total positions: {total_positions}, fine codebooks: {n_fine_codebooks}, seq_len: {seq_len}")
                    
                    # Reshape only for fine codebooks
                    vampnet_logits_fine = vampnet_logits.reshape(batch_size, vocab_size, n_fine_codebooks, seq_len)
                    vampnet_logits_fine = vampnet_logits_fine.transpose(0, 2, 3, 1)  # [batch, n_fine_cb, seq_len, vocab_size]
                    
                    # Create full logits array with zeros for conditioning codebooks
                    vampnet_logits = np.zeros((batch_size, config['n_codebooks'], seq_len, vocab_size))
                    vampnet_logits[:, config['n_conditioning_codebooks']:, :, :] = vampnet_logits_fine
                    print(f"  Reshaped to full shape: {vampnet_logits.shape}")
                else:
                    # Coarse model
                    n_codebooks = config['n_codebooks']
                    seq_len = total_positions // n_codebooks
                    
                    print(f"  Coarse model: reshaping from {vampnet_logits.shape}")
                    print(f"  Total positions: {total_positions}, codebooks: {n_codebooks}, seq_len: {seq_len}")
                    
                    vampnet_logits = vampnet_logits.reshape(batch_size, vocab_size, n_codebooks, seq_len)
                    vampnet_logits = vampnet_logits.transpose(0, 2, 3, 1)  # [batch, n_codebooks, seq_len, vocab_size]
                    print(f"  Reshaped VampNet shape: {vampnet_logits.shape}")
                
                # Check vocabulary size
                if onnx_logits.shape[-1] == 1025 and vampnet_logits.shape[-1] == 1024:
                    print(f"  ONNX has extra mask token class, truncating...")
                    onnx_logits = onnx_logits[..., :1024]
        
        # Value comparison (only on masked positions)
        # Extract logits only at masked positions
        onnx_masked_logits = onnx_logits[model_mask]
        vampnet_masked_logits = vampnet_logits[model_mask]
        
        # Statistics
        abs_diff = np.abs(onnx_masked_logits - vampnet_masked_logits)
        rel_diff = abs_diff / (np.abs(vampnet_masked_logits) + 1e-8)
        
        print(f"\nLogits comparison at masked positions:")
        print(f"  Mean absolute difference: {np.mean(abs_diff):.6f}")
        print(f"  Max absolute difference: {np.max(abs_diff):.6f}")
        print(f"  Mean relative difference: {np.mean(rel_diff):.4%}")
        print(f"  Correlation: {np.corrcoef(onnx_masked_logits.flatten(), vampnet_masked_logits.flatten())[0, 1]:.6f}")
        
        # Check if predictions would be the same
        onnx_preds = np.argmax(onnx_masked_logits, axis=-1)
        vampnet_preds = np.argmax(vampnet_masked_logits, axis=-1)
        pred_match_rate = np.mean(onnx_preds == vampnet_preds)
        print(f"  Prediction match rate: {pred_match_rate:.1%}")
        
        # Top-k accuracy
        k = 5
        onnx_topk = np.argsort(onnx_masked_logits, axis=-1)[:, -k:]
        vampnet_topk = np.argsort(vampnet_masked_logits, axis=-1)[:, -k:]
        
        # Check if top prediction is in other's top-k
        onnx_in_vampnet_topk = np.mean([onnx_preds[i] in vampnet_topk[i] for i in range(len(onnx_preds))])
        vampnet_in_onnx_topk = np.mean([vampnet_preds[i] in onnx_topk[i] for i in range(len(vampnet_preds))])
        
        print(f"  ONNX top-1 in VampNet top-{k}: {onnx_in_vampnet_topk:.1%}")
        print(f"  VampNet top-1 in ONNX top-{k}: {vampnet_in_onnx_topk:.1%}")
        
        # Speed comparison
        if onnx_time and vampnet_time:
            speedup = vampnet_time / onnx_time
            print(f"\nSpeed comparison:")
            print(f"  ONNX: {onnx_time:.3f}s")
            print(f"  VampNet: {vampnet_time:.3f}s")
            print(f"  ONNX speedup: {speedup:.1f}x")
        
        comparison_results.append({
            "name": config['name'],
            "onnx_logits": onnx_masked_logits,
            "vampnet_logits": vampnet_masked_logits,
            "mean_abs_diff": np.mean(abs_diff),
            "max_abs_diff": np.max(abs_diff),
            "correlation": np.corrcoef(onnx_masked_logits.flatten(), vampnet_masked_logits.flatten())[0, 1],
            "pred_match_rate": pred_match_rate,
            "onnx_time": onnx_time,
            "vampnet_time": vampnet_time
        })

# Visualization
if comparison_results:
    print("\nCreating visualizations...")
    
    n_models = len(comparison_results)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(comparison_results):
        # Sample some logits for visualization
        n_samples = min(100, len(result['onnx_logits']))
        sample_indices = np.random.choice(len(result['onnx_logits']), n_samples, replace=False)
        
        # Logits scatter plot
        ax = axes[idx, 0]
        ax.scatter(result['vampnet_logits'][sample_indices].flatten()[:1000], 
                  result['onnx_logits'][sample_indices].flatten()[:1000], 
                  alpha=0.5, s=1)
        ax.plot([-10, 10], [-10, 10], 'r--', alpha=0.5)
        ax.set_xlabel('VampNet Logits')
        ax.set_ylabel('ONNX Logits')
        ax.set_title(f"{result['name']} - Logits Scatter (first 1000)")
        ax.grid(True, alpha=0.3)
        
        # Distribution comparison
        ax = axes[idx, 1]
        ax.hist(result['vampnet_logits'].flatten(), bins=50, alpha=0.5, 
                label='VampNet', density=True)
        ax.hist(result['onnx_logits'].flatten(), bins=50, alpha=0.5, 
                label='ONNX', density=True)
        ax.set_xlabel('Logit Value')
        ax.set_ylabel('Density')
        ax.set_title(f"{result['name']} - Logit Distributions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error distribution
        ax = axes[idx, 2]
        errors = result['onnx_logits'] - result['vampnet_logits']
        ax.hist(errors.flatten(), bins=50, alpha=0.7)
        ax.set_xlabel('Error (ONNX - VampNet)')
        ax.set_ylabel('Count')
        ax.set_title(f"{result['name']} - Error Distribution")
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Corr: {result['correlation']:.4f}\n"
        stats_text += f"Pred Match: {result['pred_match_rate']:.1%}\n"
        stats_text += f"Mean |Error|: {result['mean_abs_diff']:.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('transformer_forward_pass_comparison.png', dpi=150)
    print("✓ Saved visualization to transformer_forward_pass_comparison.png")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if comparison_results:
    print("\nTransformer Forward Pass Comparison:")
    for result in comparison_results:
        print(f"\n{result['name']}:")
        print(f"  ✓ Correlation: {result['correlation']:.4f} {'(excellent)' if result['correlation'] > 0.99 else '(good)' if result['correlation'] > 0.95 else '(needs improvement)'}")
        print(f"  ✓ Prediction match: {result['pred_match_rate']:.1%}")
        print(f"  ✓ Mean absolute error: {result['mean_abs_diff']:.4f}")
        if result['onnx_time'] and result['vampnet_time']:
            speedup = result['vampnet_time'] / result['onnx_time']
            print(f"  ✓ ONNX speedup: {speedup:.1f}x")
    
    overall_match = all(r['correlation'] > 0.95 for r in comparison_results)
    if overall_match:
        print("\n✓ Excellent transformer forward pass match!")
        print("The ONNX models produce highly correlated outputs with VampNet.")
    else:
        print("\n⚠️  Some differences detected in transformer outputs.")
        print("This may be due to:")
        print("- Numerical precision differences")
        print("- Different implementations of attention or activations")
        print("- Weight loading discrepancies")
else:
    print("\n✗ Could not complete transformer comparison.")
    print("Please ensure all models are properly loaded.")