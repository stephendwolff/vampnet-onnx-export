#!/usr/bin/env python3
"""
Complete comparison of Transformer Forward Pass - VampNet vs ONNX V11.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
from einops import rearrange
import time

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm
from scripts.export_vampnet_transformer_v11_fixed import VampNetTransformerV11, transfer_weights_v11


def get_vampnet_classifier_output(vampnet, latents):
    """Get VampNet's classifier output for comparison."""
    x = vampnet.embedding(latents)
    x = rearrange(x, "b d n -> b n d")
    x = vampnet.transformer(x=x, x_mask=torch.ones(1, latents.shape[2], dtype=torch.bool))
    x = rearrange(x, "b n d -> b d n")
    return vampnet.classifier(x, None)


def flatten_for_correlation(vampnet_out, v11_out):
    """Flatten outputs with correct ordering for correlation."""
    vamp_flat = []
    v11_flat = []
    
    batch, n_cb, seq_len, _ = v11_out.shape
    for cb in range(n_cb):
        for pos in range(seq_len):
            vamp_start = cb * 1024
            vamp_end = (cb + 1) * 1024
            vamp_vec = vampnet_out[0, vamp_start:vamp_end, pos]
            v11_vec = v11_out[0, cb, pos, :1024]
            vamp_flat.append(vamp_vec)
            v11_flat.append(v11_vec)
    
    return torch.cat(vamp_flat), torch.cat(v11_flat)


print("="*80)
print("STEP 5: TRANSFORMER FORWARD PASS COMPARISON")
print("="*80)

# Load models
print("\n1. Loading models...")
torch.manual_seed(42)
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
remove_weight_norm(vampnet.classifier.layers[0])
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# PyTorch V11 model
model = VampNetTransformerV11()
transfer_weights_v11("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# ONNX model
print("\n2. Loading ONNX model...")
ort_session = ort.InferenceSession("vampnet_transformer_v11.onnx")

# Test cases
print("\n3. Creating test cases...")
test_cases = []

# Test case 1: Small sequence
codes1 = torch.randint(0, 1024, (1, 4, 10))
mask1 = torch.zeros((1, 4, 10), dtype=torch.bool)
mask1[:, :, 5:] = True
masked_codes1 = codes1.clone()
masked_codes1[mask1] = 1024
test_cases.append(("Small (10 tokens)", masked_codes1))

# Test case 2: Medium sequence
codes2 = torch.randint(0, 1024, (1, 4, 50))
mask2 = torch.zeros((1, 4, 50), dtype=torch.bool)
mask2[:, :, 25:] = True
masked_codes2 = codes2.clone()
masked_codes2[mask2] = 1024
test_cases.append(("Medium (50 tokens)", masked_codes2))

# Test case 3: Large sequence
codes3 = torch.randint(0, 1024, (1, 4, 100))
mask3 = torch.zeros((1, 4, 100), dtype=torch.bool)
mask3[:, :, 50:] = True
masked_codes3 = codes3.clone()
masked_codes3[mask3] = 1024
test_cases.append(("Large (100 tokens)", masked_codes3))

# Test case 4: Different masking pattern
codes4 = torch.randint(0, 1024, (1, 4, 20))
mask4 = torch.zeros((1, 4, 20), dtype=torch.bool)
# Mask every other token
mask4[:, :, ::2] = True
masked_codes4 = codes4.clone()
masked_codes4[mask4] = 1024
test_cases.append(("Alternating mask (20 tokens)", masked_codes4))

# Run comparisons
print("\n4. Running comparisons...")
results = []

for test_name, masked_codes in test_cases:
    print(f"\n{test_name}:")
    print(f"  Input shape: {masked_codes.shape}")
    
    with torch.no_grad():
        # Get latents
        latents = vampnet.embedding.from_codes(masked_codes, codec)
        print(f"  Latents shape: {latents.shape}")
        
        # Time VampNet
        start = time.time()
        vampnet_out = vampnet(latents)
        vampnet_time = time.time() - start
        
        # Time PyTorch V11
        start = time.time()
        v11_out = model(latents)
        v11_time = time.time() - start
        
        # Time ONNX
        start = time.time()
        onnx_out = ort_session.run(None, {'latents': latents.numpy()})[0]
        onnx_time = time.time() - start
        
        # Convert outputs for comparison
        # VampNet output needs to be reshaped to match V11
        seq_len = masked_codes.shape[2]
        vampnet_classifier_out = get_vampnet_classifier_output(vampnet, latents)
        
        # Compare PyTorch models
        pytorch_match = True
        total_diff = 0
        for cb in range(4):
            for pos in range(seq_len):
                vamp_start = cb * 1024
                vamp_end = (cb + 1) * 1024
                vamp_logits = vampnet_classifier_out[0, vamp_start:vamp_end, pos]
                v11_logits = v11_out[0, cb, pos, :1024]
                diff = (vamp_logits - v11_logits).abs().max()
                total_diff += diff
                if diff > 1e-4:
                    pytorch_match = False
        
        # Compare ONNX with V11
        onnx_tensor = torch.from_numpy(onnx_out)
        onnx_diff = (v11_out - onnx_tensor).abs()
        onnx_match = onnx_diff.max() < 1e-4
        
        # Compute correlations
        vamp_flat, v11_flat = flatten_for_correlation(vampnet_classifier_out, v11_out)
        pytorch_corr = np.corrcoef(vamp_flat.numpy(), v11_flat.numpy())[0, 1]
        
        onnx_flat = onnx_tensor.flatten()
        v11_all_flat = v11_out.flatten()
        onnx_corr = np.corrcoef(onnx_flat.numpy(), v11_all_flat.numpy())[0, 1]
        
        # Results
        result = {
            'name': test_name,
            'shape': masked_codes.shape,
            'vampnet_time': vampnet_time,
            'v11_time': v11_time,
            'onnx_time': onnx_time,
            'pytorch_match': pytorch_match,
            'pytorch_corr': pytorch_corr,
            'pytorch_max_diff': total_diff / (4 * seq_len),
            'onnx_match': onnx_match,
            'onnx_corr': onnx_corr,
            'onnx_max_diff': onnx_diff.max().item()
        }
        results.append(result)
        
        print(f"  VampNet time: {vampnet_time:.4f}s")
        print(f"  V11 PyTorch time: {v11_time:.4f}s")
        print(f"  ONNX time: {onnx_time:.4f}s")
        print(f"  PyTorch match: {pytorch_match} (corr: {pytorch_corr:.4f})")
        print(f"  ONNX match: {onnx_match} (corr: {onnx_corr:.4f})")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Accuracy:")
all_pytorch_match = all(r['pytorch_match'] for r in results)
all_onnx_match = all(r['onnx_match'] for r in results)
print(f"  All PyTorch tests match: {all_pytorch_match}")
print(f"  All ONNX tests match: {all_onnx_match}")

print("\n2. Performance (speedup over VampNet):")
for r in results:
    pytorch_speedup = r['vampnet_time'] / r['v11_time']
    onnx_speedup = r['vampnet_time'] / r['onnx_time']
    print(f"  {r['name']}:")
    print(f"    PyTorch V11: {pytorch_speedup:.2f}x")
    print(f"    ONNX: {onnx_speedup:.2f}x")

print("\n3. Key Findings:")
print("  ✓ VampNet uses latents as input (not codes)")
print("  ✓ All layers use MultiHeadRelativeAttention")
print("  ✓ Only layer 0 has relative position bias")
print("  ✓ Classifier uses weight normalization (must be removed for stable weights)")
print("  ✓ Output shape: [batch, codebooks, seq_len, vocab_size+1]")

print("\n4. Implementation Details:")
print("  - Input: Latents [batch, n_codebooks * latent_dim, seq_len]")
print("  - Embedding: Conv1d projection to d_model")
print("  - Transformer: 20 layers with RMSNorm, attention, FiLM, and GatedGELU FFN")
print("  - Output: 4 separate Linear projections (one per codebook)")
print("  - Mask token: Index 1024 (vocab_size)")

# Save comparison results
import json
comparison_results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'models': {
        'vampnet': 'Original VampNet',
        'v11_pytorch': 'VampNet V11 PyTorch',
        'v11_onnx': 'VampNet V11 ONNX'
    },
    'test_results': results,
    'accuracy': {
        'pytorch_perfect_match': all_pytorch_match,
        'onnx_perfect_match': all_onnx_match
    }
}

with open('transformer_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print("\n✓ Results saved to transformer_comparison_results.json")