#!/usr/bin/env python3
"""
Check VampNet's generate method to understand the sampling process.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import inspect

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm

print("Checking VampNet's generate method...")

# Load VampNet
torch.manual_seed(42)
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
remove_weight_norm(vampnet.classifier.layers[0])
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Check generate method
print("\n1. VampNet.generate signature:")
print(inspect.signature(vampnet.generate))

# Get generate method source
print("\n2. Key parameters in generate:")
try:
    generate_source = inspect.getsource(vampnet.generate)
    
    # Look for sampling parameters
    params = []
    if 'temperature' in generate_source:
        params.append('temperature')
    if 'top_k' in generate_source:
        params.append('top_k')
    if 'top_p' in generate_source:
        params.append('top_p')
    if 'typical' in generate_source:
        params.append('typical')
    if 'alpha' in generate_source:
        params.append('alpha')
    if 'return_probs' in generate_source:
        params.append('return_probs')
        
    print(f"Found parameters: {params}")
    
    # Check for specific sampling logic
    if 'sample' in generate_source:
        print("\nGenerate calls a sample method internally")
        # Find the sample call
        for line in generate_source.split('\n'):
            if 'sample' in line and '(' in line:
                print(f"  Sample call: {line.strip()}")
                break
                
except Exception as e:
    print(f"Could not get source: {e}")

# Test generate with simple input
print("\n3. Testing generate method...")

# Create test input
codes = torch.randint(0, 1024, (1, 4, 20))
mask = torch.zeros((1, 4, 20), dtype=torch.bool)
mask[:, :, 10:] = True
masked_codes = codes.clone()
masked_codes[mask] = 1024

try:
    # Try with default parameters
    print("\nTrying generate with default parameters...")
    output = vampnet.generate(
        codec=codec,
        codes=masked_codes,
        mask=mask,
        temperature=1.0,
        top_k=None,
        top_p=None
    )
    print(f"Generate output type: {type(output)}")
    print(f"Generate output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
    
except Exception as e:
    print(f"Error with generate: {e}")
    
    # Try different parameter combinations
    print("\nTrying simpler parameter set...")
    try:
        output = vampnet.generate(codec, masked_codes, mask)
        print(f"Success! Output shape: {output.shape}")
    except Exception as e2:
        print(f"Error: {e2}")

# Check for sampling utilities in VampNet
print("\n4. Checking for sampling utilities...")
vampnet_dir = Path(vampnet.__module__).parent
sampling_files = list(vampnet_dir.glob("**/sampling*.py")) + list(vampnet_dir.glob("**/*sample*.py"))
if sampling_files:
    print(f"Found sampling-related files: {[f.name for f in sampling_files]}")

# Check the VampNet module for sampling functions
from vampnet import modules
if hasattr(modules, 'sampling'):
    print("\nFound sampling module in vampnet.modules")
    sampling_funcs = [f for f in dir(modules.sampling) if not f.startswith('_')]
    print(f"Sampling functions: {sampling_funcs}")

# Try to find the actual sampling implementation
print("\n5. Looking for sampling implementation...")
try:
    # Check if there's a _sample method
    if hasattr(vampnet, '_sample'):
        print("Found _sample method")
        print(f"_sample signature: {inspect.signature(vampnet._sample)}")
    
    # Check for sample_from_logits
    if hasattr(vampnet, 'sample_from_logits'):
        print("Found sample_from_logits method")
        print(f"sample_from_logits signature: {inspect.signature(vampnet.sample_from_logits)}")
        
except Exception as e:
    print(f"Error checking methods: {e}")

print("\n✅ Analysis complete!")