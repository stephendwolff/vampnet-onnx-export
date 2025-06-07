"""
Comprehensive analysis of VampNet model structure.
This script loads actual VampNet models and analyzes their state dict keys,
layer naming patterns, and component structure.
"""

import torch
import vampnet
import os
from collections import defaultdict, OrderedDict
import re


def load_vampnet_models():
    """Load VampNet coarse and c2f models."""
    print("=== Loading VampNet Models ===\n")
    
    # Load VampNet interface
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device='cpu',
        compile=False
    )
    
    # Get the models
    coarse_model = interface.coarse
    c2f_model = interface.c2f
    
    # Unwrap if necessary
    if hasattr(coarse_model, '_orig_mod'):
        coarse_model = coarse_model._orig_mod
    if hasattr(c2f_model, '_orig_mod'):
        c2f_model = c2f_model._orig_mod
    
    return coarse_model, c2f_model, interface


def analyze_state_dict(model, model_name):
    """Analyze state dict of a model and categorize parameters."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()} MODEL")
    print(f"{'='*60}\n")
    
    state_dict = model.state_dict()
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total parameters: {len(state_dict)}")
    print(f"Total parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print all state dict keys sorted
    print(f"\n=== ALL STATE DICT KEYS ({model_name}) ===")
    for i, (name, param) in enumerate(state_dict.items()):
        print(f"{i:3d}. {name:60s} {str(param.shape):20s} {param.dtype}")
    
    # Categorize parameters
    categories = defaultdict(list)
    layer_params = defaultdict(list)
    
    for name, param in state_dict.items():
        # Extract layer number if present
        layer_match = re.search(r'layers\.(\d+)', name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            layer_params[layer_num].append((name, param))
        
        # Categorize by type
        if 'embedding' in name:
            categories['embeddings'].append((name, param))
        elif 'pos_emb' in name or 'pos_encoding' in name:
            categories['positional'].append((name, param))
        elif 'norm' in name or 'ln' in name:
            categories['normalization'].append((name, param))
        elif 'attn' in name or 'attention' in name:
            categories['attention'].append((name, param))
        elif 'mlp' in name or 'ffn' in name or 'fc' in name:
            categories['ffn'].append((name, param))
        elif 'film' in name or 'modulation' in name:
            categories['film'].append((name, param))
        elif 'classifier' in name or 'output' in name or 'head' in name:
            categories['output'].append((name, param))
        elif 'in_proj' in name:
            categories['projections'].append((name, param))
        elif 'out_proj' in name:
            categories['projections'].append((name, param))
        else:
            categories['other'].append((name, param))
    
    # Print categorized summary
    print(f"\n=== PARAMETER CATEGORIES ({model_name}) ===")
    for category, params in categories.items():
        print(f"\n{category.upper()} ({len(params)} parameters):")
        for name, param in params[:5]:  # Show first 5
            print(f"  {name:60s} {str(param.shape):20s}")
        if len(params) > 5:
            print(f"  ... and {len(params) - 5} more")
    
    # Analyze layer structure
    if layer_params:
        print(f"\n=== LAYER STRUCTURE ({model_name}) ===")
        print(f"Number of transformer layers: {len(layer_params)}")
        
        # Analyze first layer in detail
        if 0 in layer_params:
            print(f"\nLayer 0 parameters ({len(layer_params[0])} total):")
            for name, param in sorted(layer_params[0]):
                print(f"  {name:60s} {str(param.shape):20s}")
        
        # Show layer naming patterns
        print("\n=== NAMING PATTERNS ===")
        layer_0_names = [name for name, _ in layer_params.get(0, [])]
        
        # Extract component names
        components = defaultdict(set)
        for name in layer_0_names:
            # Remove layer prefix
            component_name = re.sub(r'layers\.\d+\.', '', name)
            # Categorize
            if 'norm' in component_name:
                components['normalization'].add(component_name)
            elif 'attn' in component_name:
                components['attention'].add(component_name)
            elif 'mlp' in component_name:
                components['mlp'].add(component_name)
            elif 'film' in component_name:
                components['film'].add(component_name)
            else:
                components['other'].add(component_name)
        
        for comp_type, comp_names in components.items():
            print(f"\n{comp_type}:")
            for name in sorted(comp_names):
                print(f"  {name}")
    
    return state_dict, categories, layer_params


def compare_models(coarse_dict, c2f_dict):
    """Compare coarse and c2f model structures."""
    print(f"\n{'='*60}")
    print("COMPARING COARSE AND C2F MODELS")
    print(f"{'='*60}\n")
    
    coarse_keys = set(coarse_dict.keys())
    c2f_keys = set(c2f_dict.keys())
    
    # Find differences
    only_coarse = coarse_keys - c2f_keys
    only_c2f = c2f_keys - coarse_keys
    common = coarse_keys & c2f_keys
    
    print(f"Total keys - Coarse: {len(coarse_keys)}, C2F: {len(c2f_keys)}")
    print(f"Common keys: {len(common)}")
    print(f"Only in coarse: {len(only_coarse)}")
    print(f"Only in c2f: {len(only_c2f)}")
    
    if only_coarse:
        print("\n=== Parameters only in COARSE model ===")
        for key in sorted(only_coarse)[:10]:
            print(f"  {key}")
        if len(only_coarse) > 10:
            print(f"  ... and {len(only_coarse) - 10} more")
    
    if only_c2f:
        print("\n=== Parameters only in C2F model ===")
        for key in sorted(only_c2f)[:10]:
            print(f"  {key}")
        if len(only_c2f) > 10:
            print(f"  ... and {len(only_c2f) - 10} more")
    
    # Compare shapes for common parameters
    shape_diffs = []
    for key in common:
        if coarse_dict[key].shape != c2f_dict[key].shape:
            shape_diffs.append((key, coarse_dict[key].shape, c2f_dict[key].shape))
    
    if shape_diffs:
        print(f"\n=== Shape differences for common parameters ({len(shape_diffs)}) ===")
        for key, coarse_shape, c2f_shape in shape_diffs[:10]:
            print(f"  {key:50s} Coarse: {str(coarse_shape):15s} C2F: {str(c2f_shape)}")


def analyze_model_architecture(model, model_name):
    """Analyze the model architecture by inspecting modules."""
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}\n")
    
    # Print module hierarchy
    print("=== MODULE HIERARCHY ===")
    for name, module in model.named_modules():
        if name:  # Skip root
            indent = "  " * name.count('.')
            print(f"{indent}{name}: {module.__class__.__name__}")
    
    # Analyze specific components
    print("\n=== KEY COMPONENTS ===")
    
    # Check for embeddings
    embeddings = [(n, m) for n, m in model.named_modules() if 'embedding' in n.lower()]
    if embeddings:
        print("\nEmbeddings:")
        for name, module in embeddings:
            print(f"  {name}: {module}")
    
    # Check for transformer layers
    layers = [(n, m) for n, m in model.named_modules() if re.match(r'layers\.\d+$', n)]
    if layers:
        print(f"\nTransformer Layers: {len(layers)}")
        # Analyze first layer structure
        if layers:
            print(f"\nFirst layer structure (layers.0):")
            first_layer = layers[0][1]
            for sub_name, sub_module in first_layer.named_modules():
                if sub_name:
                    print(f"    {sub_name}: {sub_module.__class__.__name__}")


def main():
    """Main analysis function."""
    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load models
    coarse_model, c2f_model, interface = load_vampnet_models()
    
    # Analyze each model
    coarse_dict, coarse_cats, coarse_layers = analyze_state_dict(coarse_model, "COARSE")
    
    if c2f_model:
        c2f_dict, c2f_cats, c2f_layers = analyze_state_dict(c2f_model, "C2F")
        # Compare models
        compare_models(coarse_dict, c2f_dict)
        # Analyze architecture
        analyze_model_architecture(c2f_model, "C2F")
    else:
        print("\n=== C2F MODEL NOT FOUND ===")
        print("The coarse_to_fine model could not be loaded.")
    
    # Analyze architecture
    analyze_model_architecture(coarse_model, "COARSE")
    
    # Summary of key findings
    print(f"\n{'='*60}")
    print("SUMMARY OF KEY FINDINGS")
    print(f"{'='*60}\n")
    
    print("1. NAMING PATTERNS:")
    print("   - Layers: 'layers.{layer_num}.{component}'")
    print("   - Normalization: Uses 'norm1', 'norm2', 'norm'")
    print("   - Attention: Uses 'attn' prefix")
    print("   - FFN/MLP: Uses 'mlp' prefix")
    print("   - Embeddings: Various embedding layers")
    
    print("\n2. KEY DIFFERENCES FROM OUR ONNX MODEL:")
    print("   - VampNet uses specific component names we need to map")
    print("   - Different embedding structure")
    print("   - May have additional components (FiLM, etc.)")
    
    print("\n3. WEIGHT TRANSFER CONSIDERATIONS:")
    print("   - Need to map VampNet naming to our structure")
    print("   - Handle different embedding organizations")
    print("   - Account for any architectural differences")


if __name__ == "__main__":
    main()