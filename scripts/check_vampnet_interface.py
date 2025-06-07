"""Check what's available in VampNet interface."""

import vampnet
import os

# Change to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load VampNet interface
interface = vampnet.interface.Interface(
    codec_ckpt="models/vampnet/codec.pth",
    coarse_ckpt="models/vampnet/coarse.pth",
    coarse2fine_ckpt="models/vampnet/c2f.pth",
    wavebeat_ckpt="models/vampnet/wavebeat.pth",
    device='cpu',
    compile=False
)

print("=== VampNet Interface Attributes ===")
for attr in dir(interface):
    if not attr.startswith('_'):
        obj = getattr(interface, attr)
        print(f"{attr}: {type(obj)}")

print("\n=== Checking coarse model ===")
if hasattr(interface, 'coarse'):
    coarse = interface.coarse
    print(f"Type: {type(coarse)}")
    if hasattr(coarse, '_orig_mod'):
        print(f"Has _orig_mod: {type(coarse._orig_mod)}")
    if hasattr(coarse, 'state_dict'):
        print("Has state_dict method")

print("\n=== Checking coarse_to_fine ===")
if hasattr(interface, 'coarse_to_fine'):
    c2f = interface.coarse_to_fine
    print(f"Type: {type(c2f)}")
    print(f"Is callable: {callable(c2f)}")
    
print("\n=== Checking other potential c2f attributes ===")
for attr in ['c2f', 'coarse2fine', 'fine']:
    if hasattr(interface, attr):
        obj = getattr(interface, attr)
        print(f"{attr}: {type(obj)}")