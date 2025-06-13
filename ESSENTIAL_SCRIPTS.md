# Essential Scripts to Keep

These scripts created your working models and must be preserved:

## Core Model Creation Scripts

### 1. Weight Transfer
- `scripts/transfer_weights_complete_v3.py` - Transfers weights from VampNet to PyTorch format
- `scripts/extract_codec_embeddings.py` - Extracts codec embeddings (if exists)

### 2. ONNX Export
- `scripts/export_complete_model_to_onnx.py` - Exports PyTorch models to ONNX
- `scripts/export_vampnet_transformer_v2.py` - Transformer model definition
- `scripts/export_working_encoder.py` - Creates the pre-padded encoder

### 3. Pipeline
- `scripts/vampnet_full_pipeline_fixed.py` - Working pipeline for audio generation

### 4. Custom Ops (if needed)
- `scripts/custom_ops/` directory - Contains custom ONNX operators

## Commands to Recreate Models

### Coarse Model:
```bash
# Transfer weights
python scripts/transfer_weights_complete_v3.py \
    --checkpoint models/vampnet/coarse.pth \
    --output models/coarse_complete_v3.pth \
    --model-type coarse

# Export to ONNX
python scripts/export_complete_model_to_onnx.py \
    --weights models/coarse_complete_v3.pth \
    --output onnx_models_fixed/coarse_complete_v3.onnx \
    --model-type coarse
```

### C2F Model:
```bash
# Transfer weights
python scripts/transfer_weights_complete_v3.py \
    --checkpoint models/vampnet/c2f.pth \
    --output models/c2f_complete_v3.pth \
    --model-type c2f

# Export to ONNX
python scripts/export_complete_model_to_onnx.py \
    --weights models/c2f_complete_v3.pth \
    --output onnx_models_fixed/c2f_complete_v3.onnx \
    --model-type c2f
```

### Encoder:
```bash
python scripts/export_working_encoder.py
```

## Model Files to Keep

### PyTorch Models (intermediate):
- `models/coarse_complete_v3.pth`
- `models/c2f_complete_v3.pth`

### ONNX Models (final):
- `onnx_models_fixed/coarse_complete_v3.onnx`
- `onnx_models_fixed/c2f_complete_v3.onnx`
- `scripts/models/vampnet_encoder_prepadded.onnx`
- `scripts/models/vampnet_codec_decoder.onnx`

## Scripts That Can Be Archived

Everything else in scripts/ can be archived, including:
- Test scripts (test_*.py)
- Old versions (export_*_v1.py, etc.)
- Analysis scripts
- Debug scripts
- Archive/ folder