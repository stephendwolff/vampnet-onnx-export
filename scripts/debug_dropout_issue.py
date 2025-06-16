#!/usr/bin/env python3
"""
Check if dropout is causing the issue.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, FeedForwardGatedGELU

# Check VampNet's dropout
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Checking VampNet dropout settings...")
print(f"VampNet dropout: {vampnet.dropout}")

# Check first layer FFN
layer0 = vampnet.transformer.layers[0]
print(f"\nLayer 0 FFN type: {type(layer0.feed_forward)}")

# Check if FFN has dropout
if hasattr(layer0.feed_forward, 'drop'):
    print(f"FFN has dropout: {layer0.feed_forward.drop}")
    print(f"Dropout probability: {layer0.feed_forward.drop.p}")

# Check our FFN
print("\n\nChecking our FFN dropout...")
our_ffn = FeedForwardGatedGELU(1280, dropout=0.1)
print(f"Our FFN dropout: {our_ffn.drop}")
print(f"Our dropout probability: {our_ffn.drop.p}")

# Test if eval mode affects it
print("\n\nTesting eval mode effect...")
our_ffn.eval()
test_input = torch.randn(1, 10, 1280)

# Run multiple times to see if output varies
outputs = []
for i in range(5):
    with torch.no_grad():
        out = our_ffn(test_input)
        outputs.append(out)

# Check if outputs are identical
all_same = all(torch.allclose(outputs[0], outputs[i]) for i in range(1, 5))
print(f"All outputs identical in eval mode: {all_same}")

# Check training mode
our_ffn.train()
outputs_train = []
for i in range(5):
    out = our_ffn(test_input)
    outputs_train.append(out)

all_same_train = all(torch.allclose(outputs_train[0], outputs_train[i]) for i in range(1, 5))
print(f"All outputs identical in train mode: {all_same_train}")

# Check attention dropout
print("\n\nChecking attention dropout...")
print(f"VampNet attention (layer 0): {type(layer0.self_attn)}")
if hasattr(layer0.self_attn, 'dropout'):
    print(f"Attention dropout: {layer0.self_attn.dropout}")

# Check if the issue is in the model being in training mode
print("\n\nChecking model training state...")
v9_model = VampNetTransformerV9()
print(f"V9 model initial training state: {v9_model.training}")

# After eval
v9_model.eval()
print(f"V9 model after eval: {v9_model.training}")

# Check all submodules
for name, module in v9_model.named_modules():
    if isinstance(module, nn.Dropout):
        print(f"  {name}: training={module.training}, p={module.p}")