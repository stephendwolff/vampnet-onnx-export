# ONNX Model Quantization Analysis Summary

## Problem Description
The codec_encoder.onnx model was failing to quantize properly, with warnings about:
1. Failed to infer data type of tensor: '/encoder/encoder.3/Relu_output_0'
2. Failed to infer data type of tensor: '/Reshape_output_0'

## Root Cause Analysis

### Issue Identified
The ONNX model was missing type information in the `value_info` field for intermediate tensors. Specifically:
- The model had 0 entries in `graph.value_info`
- Two critical tensors lacked type annotations:
  - `/encoder/encoder.3/Relu_output_0` (output of a ReLU operation)
  - `/Reshape_output_0` (output of a Reshape operation)

### Why This Matters for Quantization
- ONNX Runtime's quantization process needs to know the data types of all tensors to properly insert quantization/dequantization operations
- Without type information, the quantizer cannot determine how to handle these intermediate tensors
- This prevented proper quantization optimization

## Solution Applied

### Step 1: Shape Inference
Running ONNX's built-in shape inference successfully resolved the issue:
```python
import onnx.shape_inference
inferred_model = onnx.shape_inference.infer_shapes(model)
```

This automatically:
- Added 796 value_info entries to the model
- Correctly inferred both problematic tensors as FLOAT type
- Preserved all existing model functionality

### Step 2: Verification
After shape inference, the problematic tensors now have proper type information:
- `/encoder/encoder.3/Relu_output_0`: Type=FLOAT, Shape=['dynamic', 512, 'dynamic']
- `/Reshape_output_0`: Type=FLOAT, Shape=['dynamic', 'dynamic', 'dynamic', 'dynamic']

## Quantization Results

### Model Size Reduction
- Original model: 72.63 MB
- Quantized model: 39.26 MB
- **Size reduction: 46.0%**

### Quantization Details
The quantization successfully:
1. Converted Conv operations to ConvInteger (3 operations)
2. Added DynamicQuantizeLinear operations (3 instances)
3. Quantized weights from FLOAT to INT8:
   - 6 weight tensors converted to INT8
   - 3 bias/scale tensors as INT64
   - Remaining 20 initializers kept as FLOAT (likely biases and other non-quantizable parameters)

### Model Structure
The model contains:
- 3 Conv layers (all successfully quantized)
- 2 ReLU activations
- Complex reshaping and manipulation operations
- Total parameters: ~19 million

## Key Learnings

1. **Always run shape inference** on ONNX models before quantization, especially when converting from PyTorch
2. **Missing value_info** is a common issue when exporting models - it doesn't affect inference but blocks optimization
3. **Dynamic quantization** successfully reduced model size by ~46% while maintaining model structure
4. The warnings about "pre-processing before quantization" can be ignored for audio models - they're meant for image classification models

## Recommendations

1. **For future exports**: Use shape inference immediately after exporting from PyTorch:
   ```python
   torch.onnx.export(model, dummy_input, "model.onnx")
   model = onnx.load("model.onnx")
   model = onnx.shape_inference.infer_shapes(model)
   onnx.save(model, "model_with_shapes.onnx")
   ```

2. **For better quantization**: Consider static quantization with calibration data for potentially better performance

3. **Model optimization**: The model has many reshape/expand operations that might benefit from graph optimization before quantization

## Files Created
- `codec_encoder_fixed.onnx` - Model with proper type information
- `codec_encoder_quantized_fixed.onnx` - Successfully quantized model (46% smaller)