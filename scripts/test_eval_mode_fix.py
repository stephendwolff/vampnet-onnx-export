#!/usr/bin/env python3
"""
Test if eval mode fixes the weight normalization issue.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet

print("Testing eval mode fix...")

# Load VampNet multiple times and immediately set to eval
print("\n1. Loading with immediate eval mode:")
vampnet1 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet1.eval()

vampnet2 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet2.eval()

# Check if weights are consistent now
classifier_weight1 = vampnet1.classifier.layers[0].weight.data
classifier_weight2 = vampnet2.classifier.layers[0].weight.data

print(f"   Classifier weights match: {torch.allclose(classifier_weight1, classifier_weight2)}")
print(f"   Weight 1 [0,0,0]: {classifier_weight1[0,0,0]:.6f}")
print(f"   Weight 2 [0,0,0]: {classifier_weight2[0,0,0]:.6f}")

if not torch.allclose(classifier_weight1, classifier_weight2):
    diff = (classifier_weight1 - classifier_weight2).abs()
    print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Try removing weight norm
print("\n2. Trying to remove weight norm:")
try:
    from torch.nn.utils import remove_weight_norm
    vampnet3 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
    
    # Remove weight norm from classifier
    remove_weight_norm(vampnet3.classifier.layers[0])
    print("   Weight norm removed successfully!")
    
    # Check the weight now
    classifier_weight3 = vampnet3.classifier.layers[0].weight.data
    print(f"   Weight shape after removal: {classifier_weight3.shape}")
    print(f"   Weight [0,0,0]: {classifier_weight3[0,0,0]:.6f}")
    
    # Compare with another instance
    vampnet4 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
    remove_weight_norm(vampnet4.classifier.layers[0])
    classifier_weight4 = vampnet4.classifier.layers[0].weight.data
    
    print(f"\n   Weights match after removing norm: {torch.allclose(classifier_weight3, classifier_weight4)}")
    
except Exception as e:
    print(f"   Error removing weight norm: {e}")

# Try setting deterministic mode
print("\n3. Testing with deterministic settings:")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

vampnet5 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet5.eval()

torch.manual_seed(42)
vampnet6 = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet6.eval()

classifier_weight5 = vampnet5.classifier.layers[0].weight.data
classifier_weight6 = vampnet6.classifier.layers[0].weight.data

print(f"   Weights match with same seed: {torch.allclose(classifier_weight5, classifier_weight6)}")
print(f"   Weight 5 [0,0,0]: {classifier_weight5[0,0,0]:.6f}")
print(f"   Weight 6 [0,0,0]: {classifier_weight6[0,0,0]:.6f}")

# Check if the issue is in the loading function itself
print("\n4. Checking VampNet.load internals:")
print(f"   VampNet.load type: {type(VampNet.load)}")

# Try to find if there's a way to load without re-initializing
import inspect
load_source = inspect.getsource(VampNet.load)
if "init_weights" in load_source or "reset_parameters" in load_source:
    print("   Found weight initialization in load function!")
    
# Final test - use state_dict directly
print("\n5. Loading state dict directly:")
checkpoint = torch.load("models/vampnet/coarse.pth", map_location='cpu')

# Create a new VampNet instance
from vampnet import VampNet as VampNetClass
model_config = checkpoint.get('metadata', {})
print(f"   Model config keys: {list(model_config.keys())}")