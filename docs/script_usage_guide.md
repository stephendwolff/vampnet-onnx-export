# Script Usage Guide

This guide explains how to use the key scripts in the VampNet ONNX Export project.

## Quick Start Scripts

### 1. Export VampNet to ONNX

```bash
# Export transformer with custom operators
python scripts/export_vampnet_transformer_v2.py

# Export coarse-to-fine transformer
python scripts/export_c2f_transformer_v2.py

# Export codec (encoder/decoder)
python scripts/export_vampnet_codec_proper.py
```

### 2. Transfer Weights

```bash
# Transfer weights from VampNet checkpoints
python scripts/transfer_weights_vampnet_v2.py

# Attempt complete weight transfer (includes embeddings)
python scripts/complete_weight_transfer.py

# Verify weight transfer success
python scripts/verify_weight_transfer.py
```

### 3. Test Models

```bash
# Test complete model functionality
python scripts/test_complete_model.py

# Compare ONNX vs VampNet outputs
python scripts/compare_onnx_vs_original.py

# Test with pre-padded encoder
python scripts/vampnet_onnx_comparison_fixed.py
```

## Detailed Script Descriptions

### Export Scripts

#### `export_vampnet_transformer_v2.py`
**Purpose**: Exports VampNet transformer with ONNX-friendly custom operators

**Usage**:
```python
python scripts/export_vampnet_transformer_v2.py
```

**Output**: 
- `coarse_transformer_v2.onnx` - Coarse transformer model
- `coarse_transformer_v2.json` - Model metadata

**Options**:
- Modify `n_layers`, `d_model`, `n_heads` in script
- Change `sequence_length` (default: 100)

---

#### `export_vampnet_codec_proper.py`
**Purpose**: Exports codec encoder and decoder separately

**Usage**:
```python
python scripts/export_vampnet_codec_proper.py
```

**Output**:
- `vampnet_codec_encoder.onnx`
- `vampnet_codec_decoder.onnx`

**Note**: Encoder has fixed output size due to ONNX limitations

---

#### `export_working_encoder.py`
**Purpose**: Creates pre-padded encoder that matches VampNet tokens (97.9% accuracy)

**Usage**:
```python
python scripts/export_working_encoder.py
```

**Output**:
- `vampnet_encoder_prepadded.onnx`

**Important**: Requires audio padded to multiples of 768 samples

### Weight Transfer Scripts

#### `transfer_weights_vampnet_v2.py`
**Purpose**: Transfers transformer layer weights from VampNet

**Usage**:
```python
cd scripts
python transfer_weights_vampnet_v2.py
```

**What it transfers**:
- ✅ Attention weights (Q, K, V, O)
- ✅ Layer normalization
- ✅ Feed-forward network
- ❌ Embeddings (incompatible architecture)
- ❌ Output classifiers (dimension mismatch)

---

#### `complete_weight_transfer.py`
**Purpose**: Attempts to transfer all weights including embeddings

**Usage**:
```python
cd scripts
python complete_weight_transfer.py
```

**Output**:
- `vampnet_onnx_weights_final.pth`
- `vampnet_transformer_final.onnx`

**Known Issues**:
- Embeddings use random initialization
- Classifier dimensions don't match

### Testing Scripts

#### `test_complete_model.py`
**Purpose**: Comprehensive model testing

**Usage**:
```python
python scripts/test_complete_model.py
```

**Tests**:
- Token generation
- Mask handling
- Output range validation
- Performance benchmarking

---

#### `vampnet_onnx_comparison_fixed.py`
**Purpose**: Compare VampNet vs ONNX with pre-padded encoder

**Usage**:
```python
python scripts/vampnet_onnx_comparison_fixed.py
```

**Output**:
- `outputs/comparison_fixed/01_original.wav`
- `outputs/comparison_fixed/02_vampnet.wav`
- `outputs/comparison_fixed/03_onnx.wav`
- `outputs/comparison_fixed/spectrograms.png`

### Analysis Scripts

#### `analyze_onnx_model.py`
**Purpose**: Analyze ONNX model structure and statistics

**Usage**:
```python
python scripts/analyze_onnx_model.py path/to/model.onnx
```

**Information provided**:
- Model size
- Number of parameters
- Layer structure
- Input/output shapes

---

#### `check_model_weights.py`
**Purpose**: Compare weight statistics between VampNet and ONNX

**Usage**:
```python
python check_model_weights.py
```

**Shows**:
- Weight initialization statistics
- Parameter count comparison
- Missing weight mappings

### Pipeline Scripts

#### `vampnet_full_pipeline_fixed.py`
**Purpose**: Run complete generation pipeline

**Usage**:
```python
python scripts/vampnet_full_pipeline_fixed.py input.wav output.wav
```

**Features**:
- Handles pre-padding automatically
- Chunks long audio
- Applies proper masking

### Custom Operators

Located in `scripts/custom_ops/`:

```python
# Test all custom operators
python scripts/test_all_custom_ops.py

# Test specific operator
python scripts/test_custom_rmsnorm.py
```

## Common Workflows

### 1. Complete Export and Test

```bash
# Export models
python scripts/export_vampnet_transformer_v2.py
python scripts/export_c2f_transformer_v2.py
python scripts/export_vampnet_codec_proper.py

# Transfer weights
cd scripts
python transfer_weights_vampnet_v2.py
cd ..

# Test
python scripts/test_complete_model.py
```

### 2. Compare Generation Quality

```bash
# Run comparison with fixed encoder
python scripts/vampnet_onnx_comparison_fixed.py

# Analyze results
python scripts/analyze_onnx_model.py scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx
```

### 3. Debug Weight Transfer

```bash
# Check weight statistics
python check_model_weights.py

# Verify transfer
python scripts/verify_weight_transfer.py

# Test weighted vs unweighted
python test_weighted_vs_unweighted.py
```

## Environment Requirements

Most scripts expect to be run from the project root:
```bash
cd /path/to/vampnet-onnx-export-cleanup
python scripts/script_name.py
```

Some scripts (marked with `cd scripts`) need to be run from the scripts directory:
```bash
cd scripts
python script_name.py
```

## Model Paths

Scripts expect models in these locations:
- VampNet models: `models/vampnet/*.pth`
- Exported ONNX: `scripts/models/*.onnx`
- Weighted ONNX: `scripts/onnx_models_fixed/*.onnx`

## Common Issues

### "Model not found" errors
- Ensure VampNet models are downloaded to `models/vampnet/`
- Run export scripts before testing scripts

### "Shape mismatch" errors
- Use pre-padded encoder for matching tokens
- Ensure mask shape is `[batch, n_codebooks, seq_len]`

### "Import error" for custom ops
- Run from project root, not scripts directory
- Ensure `sys.path` includes parent directory

## Script Dependencies

```
Export Scripts
    ↓
Weight Transfer Scripts
    ↓
Testing Scripts
    ↓
Analysis Scripts
```

Always run in this order for best results.