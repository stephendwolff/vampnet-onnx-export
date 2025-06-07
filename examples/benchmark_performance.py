#!/usr/bin/env python3
"""
Benchmark VampNet ONNX models performance across different configurations.
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import VampNetONNXPipeline
from src.validation import create_onnx_session, benchmark_model


def benchmark_pipeline(pipeline, audio_lengths=[1.0, 3.0, 5.0, 10.0], num_runs=5):
    """Benchmark the complete pipeline with different audio lengths."""
    results = {}
    
    for length in audio_lengths:
        print(f"\nBenchmarking {length}s audio...")
        
        # Create test audio
        num_samples = int(length * 44100)
        test_audio = np.random.randn(2, num_samples).astype(np.float32)
        
        # Warmup
        _ = pipeline.process_audio(test_audio[:, :44100])
        
        # Time multiple runs
        times = []
        for run in range(num_runs):
            start = time.time()
            _ = pipeline.process_audio(test_audio)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.3f}s")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        real_time_factor = avg_time / length
        
        results[f"{length}s"] = {
            "length_seconds": length,
            "num_samples": num_samples,
            "times": times,
            "avg_time": avg_time,
            "std_time": std_time,
            "real_time_factor": real_time_factor,
            "can_run_realtime": real_time_factor < 1.0
        }
        
        print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Real-time factor: {real_time_factor:.2f}x")
        
    return results


def benchmark_components(model_dir):
    """Benchmark individual components."""
    model_dir = Path(model_dir)
    results = {}
    
    components = {
        "audio_processor": {
            "inputs": {"audio": np.random.randn(1, 2, 44100).astype(np.float32)},
            "description": "Audio preprocessing"
        },
        "codec_encoder": {
            "inputs": {"audio": np.random.randn(1, 1, 44100).astype(np.float32)},
            "description": "Audio to tokens"
        },
        "mask_generator": {
            "inputs": {"codes": np.random.randint(0, 1024, (1, 14, 100), dtype=np.int64)},
            "description": "Mask generation"
        },
        "transformer": {
            "inputs": {
                "codes": np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64),
                "mask": np.random.randint(0, 2, (1, 4, 100), dtype=np.int64)
            },
            "description": "Token generation"
        },
        "codec_decoder": {
            "inputs": {"codes": np.random.randint(0, 1024, (1, 14, 100), dtype=np.int64)},
            "description": "Tokens to audio"
        }
    }
    
    print("\n" + "="*60)
    print("COMPONENT BENCHMARKS")
    print("="*60)
    
    for name, config in components.items():
        model_path = model_dir / f"{name}.onnx"
        if not model_path.exists():
            print(f"\nSkipping {name}: model not found")
            continue
            
        print(f"\n{name.upper()} - {config['description']}")
        print("-" * 40)
        
        try:
            # Create session
            session = create_onnx_session(str(model_path))
            
            # Benchmark
            stats = benchmark_model(
                session, 
                config['inputs'],
                n_runs=100,
                warmup_runs=10
            )
            
            results[name] = stats
            
            # Print results
            print(f"Mean: {stats['mean_ms']:.2f} ms")
            print(f"Std:  {stats['std_ms']:.2f} ms")
            print(f"Min:  {stats['min_ms']:.2f} ms")
            print(f"Max:  {stats['max_ms']:.2f} ms")
            print(f"P95:  {stats['p95_ms']:.2f} ms")
            
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            
    return results


def compare_with_pytorch(model_dir):
    """Compare ONNX performance with PyTorch (if available)."""
    try:
        import torch
        import vampnet
        
        print("\n" + "="*60)
        print("PYTORCH VS ONNX COMPARISON")
        print("="*60)
        
        # Load PyTorch model
        print("\nLoading PyTorch VampNet...")
        pytorch_interface = vampnet.interface.Interface.default()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_interface.to(device)
        
        # TODO: Implement comparison
        print("Comparison not yet implemented")
        
    except ImportError:
        print("\nPyTorch/VampNet not available for comparison")


def save_results(results, output_path):
    """Save benchmark results to JSON."""
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark VampNet ONNX performance')
    parser.add_argument('-m', '--models', default='models', help='ONNX models directory')
    parser.add_argument('-o', '--output', default='benchmark_results.json', help='Output JSON file')
    parser.add_argument('--components', action='store_true', help='Benchmark individual components')
    parser.add_argument('--pipeline', action='store_true', help='Benchmark complete pipeline')
    parser.add_argument('--compare', action='store_true', help='Compare with PyTorch')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Default to all benchmarks if none specified
    if not (args.components or args.pipeline or args.compare):
        args.components = True
        args.pipeline = True
    
    results = {}
    
    # Component benchmarks
    if args.components:
        component_results = benchmark_components(args.models)
        results['components'] = component_results
    
    # Pipeline benchmarks
    if args.pipeline:
        try:
            print("\n" + "="*60)
            print("PIPELINE BENCHMARKS")
            print("="*60)
            
            pipeline = VampNetONNXPipeline(model_dir=args.models)
            pipeline.warmup()
            
            pipeline_results = benchmark_pipeline(pipeline, num_runs=args.num_runs)
            results['pipeline'] = pipeline_results
            
        except Exception as e:
            print(f"Error in pipeline benchmark: {e}")
    
    # PyTorch comparison
    if args.compare:
        compare_with_pytorch(args.models)
    
    # Save results
    if results:
        save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if 'pipeline' in results:
            print("\nPipeline Performance:")
            for length, data in results['pipeline'].items():
                rt_status = "✓ Real-time" if data['can_run_realtime'] else "✗ Not real-time"
                print(f"  {length}: {data['real_time_factor']:.2f}x - {rt_status}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())