#!/usr/bin/env python3
"""
Check if different checkpoints are being loaded.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet

print("Checking checkpoint loading...")

# Load VampNet multiple times
print("\n1. Loading VampNet multiple times:")
vampnet1 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet2 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

# Check if weights are the same
classifier_weight1 = vampnet1.classifier.layers[0].weight.data
classifier_weight2 = vampnet2.classifier.layers[0].weight.data

print(f"   Classifier weights match: {torch.allclose(classifier_weight1, classifier_weight2)}")
print(f"   Weight 1 [0,0,0]: {classifier_weight1[0,0,0]:.6f}")
print(f"   Weight 2 [0,0,0]: {classifier_weight2[0,0,0]:.6f}")

# Check other components
emb_weight1 = vampnet1.embedding.out_proj.weight.data
emb_weight2 = vampnet2.embedding.out_proj.weight.data
print(f"\n   Embedding weights match: {torch.allclose(emb_weight1, emb_weight2)}")

# Check if the model is in eval mode
print(f"\n2. Model states:")
print(f"   VampNet 1 training: {vampnet1.training}")
print(f"   VampNet 2 training: {vampnet2.training}")

# Put in eval mode and check again
vampnet1.eval()
vampnet2.eval()

# Check if eval mode changes weights (shouldn't)
classifier_weight1_eval = vampnet1.classifier.layers[0].weight.data
print(f"\n3. After eval mode:")
print(f"   Weights changed: {not torch.allclose(classifier_weight1, classifier_weight1_eval)}")

# Check the actual file
print(f"\n4. Checking checkpoint file:")
checkpoint = torch.load("models/vampnet/coarse.pth", map_location='cpu')
print(f"   Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # First 5 keys

# If it's a state dict
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# Check classifier weight in checkpoint
classifier_key = None
for key in state_dict.keys():
    if 'classifier' in key and 'weight' in key:
        classifier_key = key
        break

if classifier_key:
    checkpoint_weight = state_dict[classifier_key]
    print(f"\n   Found classifier weight key: {classifier_key}")
    print(f"   Checkpoint weight shape: {checkpoint_weight.shape}")
    print(f"   Checkpoint weight [0,0,0]: {checkpoint_weight[0,0,0]:.6f}")
    
    # Compare with loaded model
    loaded_weight = vampnet1.classifier.layers[0].weight.data
    print(f"   Loaded weight [0,0,0]: {loaded_weight[0,0,0]:.6f}")
    print(f"   Weights match: {torch.allclose(checkpoint_weight, loaded_weight)}")