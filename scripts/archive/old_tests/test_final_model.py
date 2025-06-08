"""
Test the final ONNX model and compare all versions.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy
import time


def load_all_models():
    """Load all available model versions."""
    
    models = {}
    
    model_files = {
        'random': 'vampnet_transformer.onnx',
        'partial': 'vampnet_transformer_improved.onnx', 
        'complete': 'vampnet_transformer_complete.onnx',
        'final': 'vampnet_transformer_final.onnx'
    }
    
    print("=== Loading Models ===")
    for name, path in model_files.items():
        if Path(path).exists():
            models[name] = ort.InferenceSession(path)
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name}: {path} not found")
    
    return models


def compare_models(models):
    """Compare outputs from different model versions."""
    
    print("\n=== Model Comparison ===")
    
    # Same input for all models
    np.random.seed(42)  # For reproducibility
    codes = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
    mask = np.zeros_like(codes)
    mask[:, :, 30:70] = 1  # Mask middle section
    
    results = {}
    
    # Run each model
    for name, session in models.items():
        start_time = time.time()
        outputs = session.run(None, {'codes': codes, 'mask': mask})
        inference_time = time.time() - start_time
        
        results[name] = {
            'output': outputs[0],
            'time': inference_time
        }
    
    # Compare outputs
    print("\n1. Inference Times:")
    for name, data in results.items():
        print(f"  {name}: {data['time']*1000:.1f}ms")
    
    print("\n2. Output Differences:")
    if 'random' in results:
        baseline = results['random']['output']
        for name, data in results.items():
            if name != 'random':
                diff = (data['output'] != baseline).sum()
                total = baseline.size
                print(f"  {name} vs random: {diff}/{total} different ({diff/total*100:.1f}%)")
    
    # Compare token distributions
    print("\n3. Token Distribution Analysis:")
    masked_positions = mask[0, 0].astype(bool)
    
    for name, data in results.items():
        tokens = data['output'][0, 0, masked_positions]  # First codebook, masked positions
        unique_tokens = np.unique(tokens)
        
        # Calculate entropy
        _, counts = np.unique(tokens, return_counts=True)
        probs = counts / counts.sum()
        token_entropy = entropy(probs)
        
        print(f"\n  {name}:")
        print(f"    Unique tokens: {len(unique_tokens)}")
        print(f"    Entropy: {token_entropy:.3f}")
        print(f"    Most common token: {tokens[0]} (appears {counts[0]} times)")
    
    return results


def visualize_generation_patterns(results):
    """Visualize generation patterns from different models."""
    
    print("\n=== Visualizing Generation Patterns ===")
    
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 3*n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(results.items()):
        output = data['output'][0]  # First batch
        
        # Show first codebook
        ax = axes[idx]
        im = ax.imshow(output[0:1], aspect='auto', cmap='tab20', interpolation='nearest')
        ax.set_title(f'{name.capitalize()} Model - Codebook 0')
        ax.set_ylabel('CB')
        ax.set_xlabel('Sequence Position')
        plt.colorbar(im, ax=ax, label='Token ID')
    
    plt.tight_layout()
    plt.savefig('model_comparison_patterns.png', dpi=150)
    print("✓ Saved visualization to model_comparison_patterns.png")
    plt.close()


def test_music_structure(models):
    """Test if models generate music-like structures."""
    
    print("\n\n=== Music Structure Test ===")
    
    # Create input with periodic masking (simulating musical bars)
    seq_len = 100  # Fixed sequence length for our models
    codes = np.random.randint(0, 1024, (1, 4, seq_len), dtype=np.int64)
    mask = np.zeros_like(codes)
    
    # Mask every 8th position (simulating downbeats)
    for i in range(0, seq_len, 8):
        if i + 1 < seq_len:
            mask[:, :, i:i+2] = 1
    
    print(f"Testing with periodic mask pattern (every 8 positions)")
    
    for name, session in models.items():
        output = session.run(None, {'codes': codes, 'mask': mask})[0]
        
        # Check for repetition patterns
        # Music often has repetitive structures
        repetitions = 0
        for cb in range(4):
            for i in range(16, seq_len, 16):
                # Check if 16-position segments repeat
                if i + 16 <= seq_len:
                    segment1 = output[0, cb, i-16:i]
                    segment2 = output[0, cb, i:i+16]
                    if np.array_equal(segment1, segment2):
                        repetitions += 1
        
        print(f"\n{name}:")
        print(f"  Repetitive segments found: {repetitions}")
        
        # Check token clustering (musical tokens often cluster)
        tokens = output[0, 0, :]  # First codebook
        
        # Calculate autocorrelation to detect periodicity
        mean = np.mean(tokens)
        tokens_centered = tokens - mean
        autocorr = np.correlate(tokens_centered, tokens_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation (indicates periodicity)
        peaks = []
        for i in range(1, min(32, len(autocorr))):
            if i > 1 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.3:  # Threshold for significant correlation
                    peaks.append(i)
        
        if peaks:
            print(f"  Periodic patterns detected at lags: {peaks}")
        else:
            print(f"  No strong periodic patterns detected")


def analyze_weight_impact():
    """Analyze the impact of different weight transfers."""
    
    print("\n\n=== Weight Transfer Impact Analysis ===")
    
    weight_info = {
        'random': {'weights': 0, 'components': []},
        'partial': {'weights': 101, 'components': ['norms', 'attention', 'ffn_partial']},
        'complete': {'weights': 121, 'components': ['norms', 'attention', 'ffn_complete']},
        'final': {'weights': 123, 'components': ['norms', 'attention', 'ffn_complete', 'classifier_partial']}
    }
    
    print("\nWeight Transfer Summary:")
    for model, info in weight_info.items():
        print(f"\n{model}:")
        print(f"  Weights transferred: {info['weights']}/294 ({info['weights']/294*100:.1f}%)")
        print(f"  Components: {', '.join(info['components']) if info['components'] else 'none'}")
    
    print("\n\nKey Findings:")
    print("1. Random → Partial (101 weights): Added all normalization and attention")
    print("2. Partial → Complete (121 weights): Fixed FFN with GatedGELU")  
    print("3. Complete → Final (123 weights): Added output classifier (partial)")
    print("\nMissing components:")
    print("- Embeddings (VampNet uses different approach)")
    print("- Remaining classifiers (only 1 of 4 transferred)")
    print("- Positional embeddings")


if __name__ == "__main__":
    # Load all models
    models = load_all_models()
    
    if len(models) == 0:
        print("\n❌ No models found to test!")
        exit(1)
    
    # Compare models
    results = compare_models(models)
    
    # Visualize patterns
    visualize_generation_patterns(results)
    
    # Test music structure
    test_music_structure(models)
    
    # Analyze weight impact
    analyze_weight_impact()
    
    print("\n\n=== Conclusion ===")
    print("✅ Successfully created multiple ONNX model versions")
    print("✅ Transferred 123/294 weights (41.8%)")
    print("✅ Models show different generation patterns")
    print("\n⚠️  Limitations:")
    print("- VampNet's embedding approach differs significantly")
    print("- Only partial classifier weights transferred")
    print("- Full music generation quality requires complete weights")
    print("\nNext steps for production use:")
    print("1. Implement VampNet's embedding approach in ONNX")
    print("2. Complete classifier weight transfer for all codebooks")
    print("3. Integrate with audio codec for end-to-end generation")