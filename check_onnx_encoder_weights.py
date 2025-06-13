#!/usr/bin/env python3
"""Check if ONNX encoder has proper weights or random initialization."""

import onnx
import numpy as np
from pathlib import Path

def check_encoder_weights(onnx_path):
    """Analyze ONNX encoder weights."""
    print(f"Checking ONNX encoder: {onnx_path}")
    
    # Load model
    model = onnx.load(onnx_path)
    
    # Get all initializers (weights)
    initializers = {init.name: init for init in model.graph.initializer}
    print(f"\nFound {len(initializers)} weight tensors")
    
    # Analyze weights
    weight_stats = []
    for name, init in initializers.items():
        # Convert to numpy
        weights = onnx.numpy_helper.to_array(init)
        
        # Calculate statistics
        stats = {
            'name': name,
            'shape': weights.shape,
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'zeros': np.sum(weights == 0),
            'size': weights.size
        }
        weight_stats.append(stats)
    
    # Print analysis
    print("\nWeight Analysis:")
    print("-" * 80)
    
    # Look for signs of random initialization
    suspicious_weights = []
    
    for stats in weight_stats:
        name = stats['name']
        print(f"\n{name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Check for suspicious patterns
        if 'embedding' in name.lower() or 'codebook' in name.lower():
            print("  ⚠️  This looks like a codebook/embedding layer")
            
            # Random embeddings often have mean near 0 and std near 0.02-0.1
            if abs(stats['mean']) < 0.01 and 0.01 < stats['std'] < 0.15:
                print("  ❌ WARNING: Statistics suggest random initialization!")
                suspicious_weights.append(name)
            elif stats['zeros'] > stats['size'] * 0.1:
                print("  ❌ WARNING: Many zeros suggest incomplete initialization!")
                suspicious_weights.append(name)
                
    # Look for VampNet-specific layer names
    print("\n" + "=" * 80)
    print("Looking for VampNet codec layers:")
    
    vampnet_layers = []
    for name in initializers.keys():
        if any(pattern in name for pattern in ['quantizer', 'codec', 'vq', 'codebook']):
            vampnet_layers.append(name)
            
    if vampnet_layers:
        print(f"Found {len(vampnet_layers)} potential codec layers:")
        for layer in vampnet_layers:
            print(f"  - {layer}")
    else:
        print("❌ No codec-specific layers found! The model might not contain codec weights.")
        
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    if suspicious_weights:
        print(f"❌ Found {len(suspicious_weights)} layers with suspicious weight patterns:")
        for name in suspicious_weights:
            print(f"   - {name}")
        print("\nThe ONNX model likely has random weights instead of trained codec weights!")
    else:
        print("✅ Weight patterns look reasonable (but manual verification recommended)")
        
    return suspicious_weights


if __name__ == "__main__":
    # Check the pre-padded encoder
    encoder_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    if encoder_path.exists():
        suspicious = check_encoder_weights(encoder_path)
        
        if suspicious:
            print("\n" + "!" * 80)
            print("RECOMMENDATION: The encoder needs to be re-exported with proper weight transfer!")
            print("The codec weights were not properly included during ONNX export.")
            print("!" * 80)
    else:
        print(f"Encoder not found at {encoder_path}")