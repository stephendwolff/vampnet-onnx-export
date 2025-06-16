#!/usr/bin/env python3
"""
Final corrected mask test showing the exact issue and fix.

The key insights:
1. VampNet uses 1 = MASKED, 0 = UNMASKED
2. codebook_mask(mask, val1) sets mask[:, val1:, :] = 1 (masks codebooks >= val1)
3. With rand_mask_intensity=0.0 and periodic_prompt=7:
   - Initial mask is all zeros (nothing masked)
   - Periodic mask has zeros at positions 0,7,14... 
   - AND operation keeps all zeros
   - codebook_mask then sets codebooks >= 3 to 1
"""

import os
import sys
import numpy as np
import torch
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vampnet.mask import linear_random, periodic_mask, mask_and, codebook_mask

print("=" * 80)
print("VAMPNET MASK GENERATION - EXACT REPLICATION")
print("=" * 80)

# Create test tensor
x = torch.zeros(1, 14, 100)  # Use 100 tokens like in the actual test

# Test case: rand_mask_intensity=0.0, periodic_prompt=7, upper_codebook_mask=3
print("\nTest case: rand_mask_intensity=0.0, periodic_prompt=7, upper_codebook_mask=3")
print("-" * 80)

# Step 1: linear_random with intensity 0.0
torch.manual_seed(42)
mask_step1 = linear_random(x, torch.tensor(0.0))
print(f"Step 1 - linear_random(0.0):")
print(f"  Codebook 0: {mask_step1[0, 0, :20].tolist()}")
print(f"  All zeros? {torch.all(mask_step1 == 0).item()}")

# Step 2: periodic_mask
mask_step2 = periodic_mask(x, period=7, width=1, random_roll=True)
print(f"\nStep 2 - periodic_mask(7, random_roll=True):")
print(f"  Codebook 0: {mask_step2[0, 0, :20].tolist()}")
# Find where zeros are
zeros = torch.where(mask_step2[0, 0, :] == 0)[0]
print(f"  Zero positions: {zeros[:10].tolist()}...")

# Step 3: mask_and
mask_step3 = mask_and(mask_step1, mask_step2)
print(f"\nStep 3 - mask_and(step1, step2):")
print(f"  Codebook 0: {mask_step3[0, 0, :20].tolist()}")
print(f"  All zeros? {torch.all(mask_step3 == 0).item()}")

# Step 4: codebook_mask
mask_final = codebook_mask(mask_step3, val1=3)
print(f"\nStep 4 - codebook_mask(step3, val1=3):")
print(f"  Codebook 0: {mask_final[0, 0, :20].tolist()}")
print(f"  Codebook 2: {mask_final[0, 2, :20].tolist()}")
print(f"  Codebook 3: {mask_final[0, 3, :20].tolist()}")
print(f"  Codebook 13: {mask_final[0, 13, :20].tolist()}")

# Check the pattern
print(f"\nFinal mask pattern:")
for cb in [0, 2, 3, 13]:
    ratio = torch.mean(mask_final[0, cb, :].float()).item()
    print(f"  Codebook {cb}: {ratio:.1%} masked")

print("\nExpected behavior:")
print("  - Codebooks 0-2: 0% masked (below upper_codebook_mask)")
print("  - Codebooks 3-13: 100% masked (at or above upper_codebook_mask)")

# Now show the correct ONNX implementation
print("\n" + "=" * 80)
print("CORRECT ONNX IMPLEMENTATION")
print("=" * 80)

def build_mask_fixed(
    z: np.ndarray,
    rand_mask_intensity: float = 1.0,
    periodic_prompt: int = 0,
    upper_codebook_mask: int = 0,
    random_roll: bool = True,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Fixed mask generator that exactly matches VampNet's behavior.
    
    Key fix: codebook_mask sets codebooks >= upper_codebook_mask to 1 (masked),
    not codebooks < upper_codebook_mask.
    """
    shape = z.shape
    batch_size, n_codebooks, seq_len = shape
    
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: linear_random
    if rand_mask_intensity <= 0:
        mask = np.zeros(shape, dtype=np.int64)
    elif rand_mask_intensity >= 1:
        mask = np.ones(shape, dtype=np.int64)
    else:
        mask = (np.random.rand(*shape) < rand_mask_intensity).astype(np.int64)
    
    # Step 2: periodic_mask
    if periodic_prompt > 0:
        periodic_mask = np.ones(shape, dtype=np.int64)
        
        if random_roll:
            offset = np.random.randint(0, periodic_prompt)
        else:
            offset = 0
        
        # Set periodic positions to 0 (unmasked)
        for i in range(offset, seq_len, periodic_prompt):
            periodic_mask[:, :, i] = 0
        
        # AND operation (minimum)
        mask = np.minimum(mask, periodic_mask)
    
    # Step 3: codebook_mask - this is the key fix!
    if upper_codebook_mask > 0:
        # Set codebooks >= upper_codebook_mask to 1 (masked)
        mask[:, upper_codebook_mask:, :] = 1
    
    return mask.astype(bool)

# Test the fixed implementation
np.random.seed(42)
onnx_mask = build_mask_fixed(
    x.numpy(),
    rand_mask_intensity=0.0,
    periodic_prompt=7,
    upper_codebook_mask=3,
    random_roll=True,
    seed=42
)

print("\nONNX mask (fixed):")
print(f"  Codebook 0: {onnx_mask[0, 0, :20].astype(int)}")
print(f"  Codebook 2: {onnx_mask[0, 2, :20].astype(int)}")
print(f"  Codebook 3: {onnx_mask[0, 3, :20].astype(int)}")
print(f"  Codebook 13: {onnx_mask[0, 13, :20].astype(int)}")

# Check if they match
vampnet_np = mask_final.numpy()
match = np.array_equal(vampnet_np, onnx_mask)
overlap = np.mean(vampnet_np == onnx_mask)

print(f"\nComparison:")
print(f"  Exact match: {match}")
print(f"  Overlap: {overlap:.1%}")
print(f"  VampNet mask ratio: {np.mean(vampnet_np):.1%}")
print(f"  ONNX mask ratio: {np.mean(onnx_mask):.1%}")

if not match:
    # Find differences
    diff_positions = np.where(vampnet_np != onnx_mask)
    n_diffs = len(diff_positions[0])
    print(f"  Number of differences: {n_diffs}")
    if n_diffs < 20:
        for i in range(min(5, n_diffs)):
            b, c, t = diff_positions[0][i], diff_positions[1][i], diff_positions[2][i]
            print(f"    Diff at [{b},{c},{t}]: VampNet={vampnet_np[b,c,t]}, ONNX={onnx_mask[b,c,t]}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The issue was that the ONNX implementation had the codebook_mask logic inverted!")
print("VampNet: mask[:, upper_codebook_mask:, :] = 1  (masks codebooks >= threshold)")
print("ONNX was: mask[:, :upper_codebook_mask, :] = 1  (masks codebooks < threshold)")
print("\nWith this fix, ONNX now correctly produces mostly zeros for lower codebooks")
print("and mostly ones for upper codebooks, matching VampNet's behavior.")