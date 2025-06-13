#!/usr/bin/env python3
"""Check which codec VampNet actually uses."""

import vampnet
import torch

# Load VampNet
interface = vampnet.interface.Interface(
    codec_ckpt="models/vampnet/codec.pth",
    coarse_ckpt="models/vampnet/coarse.pth",
    coarse2fine_ckpt="models/vampnet/c2f.pth",
    wavebeat_ckpt="models/vampnet/wavebeat.pth",
    device='cpu'
)

print("VampNet codec information:")
print(f"Codec type: {type(interface.codec)}")
print(f"Codec module: {interface.codec.__class__.__module__}")
print(f"Codec class: {interface.codec.__class__.__name__}")

# Check if it's DAC
if hasattr(interface.codec, 'model_type'):
    print(f"Model type: {interface.codec.model_type}")

# Check attributes
print("\nCodec attributes:")
attrs = ['sample_rate', 'hop_length', 'n_codebooks', 'model_type', 'model_bitrate']
for attr in attrs:
    if hasattr(interface.codec, attr):
        print(f"  {attr}: {getattr(interface.codec, attr)}")

# Check the actual model structure
print("\nChecking model structure...")
if hasattr(interface.codec, 'encoder'):
    print(f"Encoder type: {type(interface.codec.encoder)}")
if hasattr(interface.codec, 'decoder'):
    print(f"Decoder type: {type(interface.codec.decoder)}")
if hasattr(interface.codec, 'quantizer'):
    print(f"Quantizer type: {type(interface.codec.quantizer)}")