# VampNet ONNX Export

This module provides utilities for exporting VampNet components to ONNX format, enabling deployment on various platforms and inference frameworks.

## Overview

VampNet is a masked acoustic token modeling approach for music generation. This ONNX export module breaks down the VampNet pipeline into modular components that can be exported, optimized, and deployed independently.

### Components

1. **Audio Processor** - Handles audio preprocessing (resampling, normalization, padding)
2. **Codec Encoder** - Converts audio waveforms to discrete tokens
3. **Mask Generator** - Creates masking patterns for token generation
4. **Transformer** - Generates new tokens in masked positions
5. **Codec Decoder** - Converts tokens back to audio waveforms

## Installation

```bash
# Install dependencies
pip install torch onnx onnxruntime audiotools vampnet

# descript-audiotools @ git+https://github.com/hugofloresgarcia/audiotools.git

# For optimization features
pip install onnxruntime-tools
```

## Quick Start

### 1. Basic Export

```python
from vampnet_onnx import export_all_components

# Export all components with default settings
exported_models = export_all_components(
    output_dir="onnx_models",
    codec_encoder={'use_simplified': True},
    codec_decoder={'use_simplified': True},
    transformer={'use_simplified': True}
)
```

### 2. Using the ONNX Pipeline

```python
from vampnet_onnx import VampNetONNXPipeline
import numpy as np

# Initialize pipeline
pipeline = VampNetONNXPipeline(model_dir="onnx_models")

# Process audio
audio = np.random.randn(2, 44100)  # 1 second stereo audio
results = pipeline.process_audio(
    audio,
    sample_rate=44100,
    periodic_prompt=7,
    upper_codebook_mask=3
)

output_audio = results['output_audio']
```

### 3. Component-by-Component Usage

```python
import torch
from vampnet_onnx import (
    AudioProcessor, CodecEncoder, MaskGenerator,
    TransformerWrapper, CodecDecoder
)

# Create individual components
audio_processor = AudioProcessor()
encoder = CodecEncoder(n_codebooks=14, vocab_size=1024)
mask_gen = MaskGenerator(n_codebooks=14)
transformer = TransformerWrapper(n_codebooks=4)
decoder = CodecDecoder(n_codebooks=14)

# Create example inputs for each component
audio_input = torch.randn(1, 2, 44100)  # [batch, channels, samples]
processed_audio = torch.randn(1, 1, 44100)  # [batch, 1, samples]
codes = torch.randint(0, 1024, (1, 14, 100))  # [batch, codebooks, sequence]
coarse_codes = torch.randint(0, 1024, (1, 4, 100))  # [batch, 4, sequence]

# Export each component
torch.onnx.export(audio_processor, audio_input, "audio_processor.onnx")
torch.onnx.export(encoder, processed_audio, "encoder.onnx")
torch.onnx.export(mask_gen, (codes, 7, 3, 0), "mask_generator.onnx")
torch.onnx.export(transformer, coarse_codes, "transformer.onnx")
torch.onnx.export(decoder, codes, "decoder.onnx")
```

## Testing and Validation

### 1. Validate ONNX Models

```python
from vampnet_onnx import validate_onnx_model, analyze_model_size

# Validate model structure
validate_onnx_model("onnx_models/transformer.onnx")

# Analyze model size and complexity
analyze_model_size("onnx_models/transformer.onnx")
```

### 2. Compare with PyTorch

```python
import torch
from vampnet_onnx import validate_vampnet_component, TransformerWrapper

# Create model and example inputs
transformer = TransformerWrapper(n_codebooks=4)
codes = torch.randint(0, 1024, (1, 4, 100))
mask = torch.randint(0, 2, (1, 4, 100))

# Compare PyTorch vs ONNX outputs
validate_vampnet_component(
    pytorch_model=transformer,
    onnx_path="transformer.onnx",
    example_inputs=(codes, mask),
    input_names=['codes', 'mask']
)
```

### 3. Benchmark Performance

```python
import numpy as np
from vampnet_onnx import create_onnx_session, benchmark_model

# Create session
session = create_onnx_session("transformer.onnx")

# Create numpy arrays for benchmarking
codes_array = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
mask_array = np.random.randint(0, 2, (1, 4, 100), dtype=np.int64)

# Benchmark
stats = benchmark_model(
    session,
    inputs={'codes': codes_array, 'mask': mask_array},
    n_runs=100
)
```

### 4. Run Complete Tests

Use the provided notebooks for comprehensive testing:

```bash
# End-to-end pipeline testing
jupyter notebook vampnet_onnx_pipeline_test.ipynb

# Optimization and quantization
jupyter notebook vampnet_onnx_optimization.ipynb
```

## Optimization

### 1. Graph Optimization

```python
import onnxruntime as ort

# Optimize during session creation
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "models/transformer_opt.onnx"

# Create session (this will save the optimized model)
session = ort.InferenceSession("models/transformer.onnx", sess_options)
```

### 2. Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (easier, good for CPU)
quantize_dynamic(
    "models/transformer_opt.onnx",
    "models/transformer_int8.onnx",
    weight_type=QuantType.QInt8
)
```

### 3. Production Pipeline

```python
# Example: Create optimized models for deployment
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_for_production(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # List of models to optimize
    models = ['audio_processor', 'codec_encoder', 'mask_generator', 
              'transformer', 'codec_decoder']
    
    for model in models:
        input_path = f"{input_dir}/{model}.onnx"
        output_path = f"{output_dir}/{model}_opt.onnx"
        
        if os.path.exists(input_path):
            # Apply dynamic quantization
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QInt8
            )
            print(f"Optimized {model}")

# Run optimization
optimize_for_production("onnx_models", "onnx_models_production")
```

## Deployment Examples

### ONNX Runtime (Python)

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession("transformer.onnx")

# Prepare inputs
codes_numpy = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
mask_numpy = np.random.randint(0, 2, (1, 4, 100), dtype=np.int64)

# Run inference
outputs = session.run(None, {
    'codes': codes_numpy,
    'mask': mask_numpy
})
```

### ONNX Runtime (C++)

```cpp
Ort::Session session(env, "transformer.onnx", session_options);
auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names.data(), input_tensors.data(), input_names.size(),
    output_names.data(), output_names.size()
);
```

### Web Deployment (ONNX.js)

```javascript
const session = await onnx.InferenceSession.create('transformer.onnx');
const outputs = await session.run({
    'codes': codesTensor,
    'mask': maskTensor
});
```

## Limitations

1. **Simplified Codec**: Current implementation uses a simplified codec instead of the full DAC codec
2. **Deterministic Generation**: ONNX export uses argmax instead of sampling for token generation
3. **Fixed Shapes**: Some models may require fixed batch sizes for export
4. **Model Size**: Full VampNet models are large (300M+ parameters) and may need quantization

## TODOs and Next Steps

### High Priority
- [ ] Implement full DAC codec export support
- [ ] Add proper sampling/temperature support in ONNX
- [ ] Create streaming inference support for real-time applications
- [ ] Add mobile-specific optimizations (Core ML, TFLite conversion)

### Medium Priority
- [ ] Implement iterative refinement in ONNX (multiple generation passes)
- [ ] Add support for conditional generation (style tokens)
- [ ] Create WebAssembly build for browser deployment
- [ ] Implement batch processing optimizations

### Low Priority
- [ ] Add model pruning support
- [ ] Create platform-specific optimizations (TensorRT, CoreML)
- [ ] Build C++ inference library
- [ ] Add support for model ensembling

### Research & Development
- [ ] Investigate knowledge distillation for smaller models
- [ ] Explore neural architecture search for efficient variants
- [ ] Test deployment on edge devices (Raspberry Pi, Jetson)
- [ ] Benchmark against other audio generation models

## Troubleshooting

### Common Issues

1. **"Model too large for export"**
   - Use quantization or split into smaller components
   - Increase proto size limit: `onnx.checker.check_model(model, full_check=True)`

2. **"Unsupported operator"**
   - Check ONNX opset version compatibility
   - Implement custom operators if needed
   - Use TorchScript as intermediate format

3. **"Shape inference failed"**
   - Ensure all tensor shapes are defined
   - Use dynamic axes for variable-length inputs
   - Add explicit shape annotations

4. **"Performance slower than PyTorch"**
   - Enable graph optimizations
   - Use appropriate execution providers (CUDA, TensorRT)
   - Apply quantization for CPU inference

## Contributing

Contributions are welcome! Please focus on:
- Improving codec export compatibility
- Adding platform-specific optimizations
- Creating deployment examples
- Benchmarking and performance improvements

## License

Same as VampNet project license.

## References

- [VampNet Paper](https://arxiv.org/abs/2307.04686)
- [ONNX Documentation](https://onnx.ai/docs/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [VampNet GitHub](https://github.com/hugofloresgarcia/vampnet)