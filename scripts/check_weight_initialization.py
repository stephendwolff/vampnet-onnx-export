#!/usr/bin/env python3
"""
Check if weights are being re-initialized after transfer.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# First, let's check what happens when we create a Linear layer
print("1. Default Linear layer initialization:")
linear = nn.Linear(1280, 1025)
print(f"   Initial weight stats: mean={linear.weight.data.mean():.6f}, std={linear.weight.data.std():.6f}")

# Now check V10 model initialization
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10

print("\n2. V10 model initialization:")
model = VampNetTransformerV10(n_layers=1)

# Check output projections
for i in range(4):
    w = model.output_projs[i].weight.data
    print(f"   Proj {i} weight stats: mean={w.mean():.6f}, std={w.std():.6f}")

# Check if the model is doing any post-init
print("\n3. Checking for post-initialization hooks:")
for name, module in model.named_modules():
    if hasattr(module, '_forward_pre_hooks'):
        if len(module._forward_pre_hooks) > 0:
            print(f"   {name} has forward pre-hooks")
    if hasattr(module, '_forward_hooks'):
        if len(module._forward_hooks) > 0:
            print(f"   {name} has forward hooks")

# Let's trace through the transfer function step by step
print("\n4. Tracing transfer_weights_v10:")

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
codec = DAC.load(Path("models/vampnet/codec.pth")) 

# Get classifier weights
classifier_weight = vampnet.classifier.layers[0].weight.data.squeeze(-1)
classifier_bias = vampnet.classifier.layers[0].bias.data

# Create fresh model
model2 = VampNetTransformerV10(n_layers=1)

# Check weights before transfer
print("\n   Before transfer:")
print(f"   Proj 0 weight[0,0]: {model2.output_projs[0].weight.data[0,0]:.6f}")

# Transfer just the classifier weights manually
for i in range(4):
    vamp_start = i * 1024
    vamp_end = (i + 1) * 1024
    
    # Transfer weights
    model2.output_projs[i].weight.data[:1024] = classifier_weight[vamp_start:vamp_end]
    model2.output_projs[i].weight.data[1024] = 0
    
    # Transfer bias
    model2.output_projs[i].bias.data[:1024] = classifier_bias[vamp_start:vamp_end]
    model2.output_projs[i].bias.data[1024] = 0

# Check weights after manual transfer
print("\n   After manual transfer:")
print(f"   Proj 0 weight[0,0]: {model2.output_projs[0].weight.data[0,0]:.6f}")
print(f"   VampNet weight[0,0]: {classifier_weight[0,0]:.6f}")
print(f"   Match: {model2.output_projs[0].weight.data[0,0] == classifier_weight[0,0]}")

# Now use the full transfer function
from scripts.export_vampnet_transformer_v10_all_relative import transfer_weights_v10

model3 = VampNetTransformerV10(n_layers=1)
transfer_weights_v10("models/vampnet/coarse.pth", model3, "models/vampnet/codec.pth")

print("\n   After transfer_weights_v10:")
print(f"   Proj 0 weight[0,0]: {model3.output_projs[0].weight.data[0,0]:.6f}")
print(f"   VampNet weight[0,0]: {classifier_weight[0,0]:.6f}")
print(f"   Match: {torch.allclose(model3.output_projs[0].weight.data[0,0], classifier_weight[0,0])}")

# Check if something happened to other layers
print("\n5. Checking other layer weights:")
print(f"   Embedding weight changed: {not torch.allclose(model2.embedding.out_proj.weight.data, model3.embedding.out_proj.weight.data)}")
print(f"   Layer 0 norm weight changed: {not torch.allclose(model2.layers[0]['norm_1'].weight.data, model3.layers[0]['norm_1'].weight.data)}")