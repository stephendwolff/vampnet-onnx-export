# File Structure Guide

This guide explains the purpose and functionality of each file in the VampNet ONNX Export project.

## Core Library (`vampnet_onnx/`)

### Main Components

- **`__init__.py`** - Package initialization, exports main classes
- **`audio_processor.py`** - Audio preprocessing (resampling, normalization, padding)
- **`codec_wrapper.py`** - Wrappers for ONNX codec encoder/decoder models
- **`exporters.py`** - Main export functions for converting VampNet to ONNX
- **`mask_generator.py`** - Creates masking patterns for token generation
- **`mask_generator_onnx.py`** - ONNX-compatible mask generation
- **`pipeline.py`** - End-to-end pipeline chaining all components
- **`transformer_wrapper.py`** - Wrapper for ONNX transformer models
- **`validation.py`** - Tools for validating ONNX models against PyTorch
- **`vampnet_codec.py`** - Codec implementation details
- **`torchscript_utils.py`** - Utilities for TorchScript export (alternative to ONNX)

## Scripts (`scripts/`)

### Current/Active Scripts

#### Export Scripts
- **`export_vampnet_transformer_v2.py`** ⭐ - Current transformer export with custom ops
- **`export_c2f_transformer_v2.py`** ⭐ - Exports coarse-to-fine transformer
- **`export_vampnet_codec_proper.py`** ⭐ - Exports codec encoder/decoder
- **`export_working_encoder.py`** - Creates pre-padded encoder (97.9% token match)

#### Weight Transfer Scripts
- **`transfer_weights_vampnet_v2.py`** ⭐ - Transfers weights with VampNet naming
- **`complete_weight_transfer.py`** ⭐ - Attempts complete weight transfer (including embeddings)
- **`verify_weight_transfer.py`** - Verifies if weights transferred correctly

#### Testing Scripts
- **`test_complete_model.py`** - Tests the complete exported model
- **`test_transformer_inference.py`** - Tests transformer generation behavior
- **`test_all_custom_ops.py`** - Validates all custom ONNX operators
- **`test_custom_rmsnorm.py`** - Tests RMSNorm implementation
- **`test_full_export_pipeline.py`** - End-to-end export testing
- **`test_vampnet_generation.py`** - Tests generation quality
- **`test_weighted_models.py`** - Compares weighted vs unweighted models
- **`test_quantization_fixed.py`** - Tests quantized model performance

#### Analysis Scripts
- **`analyze_onnx_model.py`** - Analyzes ONNX model structure and size
- **`analyze_quantization_detailed.py`** - Detailed quantization analysis
- **`check_vampnet_interface.py`** - Checks VampNet API compatibility
- **`compare_onnx_vs_original.py`** - Compares ONNX vs PyTorch outputs

#### Pipeline Scripts
- **`vampnet_full_pipeline_fixed.py`** ⭐ - Current working pipeline implementation
- **`vampnet_onnx_complete_demo.py`** - Complete demo with all components
- **`vampnet_onnx_comparison_fixed.py`** - Fixed comparison handling pre-padded encoder

#### Cleanup/Utility Scripts
- **`cleanup_models.py`** - Manual model cleanup
- **`cleanup_models_auto.py`** - Automatic model directory cleanup
- **`fix_onnx_types.py`** - Fixes ONNX type inference issues

### Custom Operators (`scripts/custom_ops/`)

- **`rmsnorm_onnx.py`** - RMSNorm layer for ONNX
- **`film_onnx.py`** - FiLM (Feature-wise Linear Modulation) conditioning
- **`multihead_attention_onnx.py`** - Custom attention for dynamic shapes
- **`codebook_embedding_onnx.py`** - Embedding layer for multiple codebooks

### Archived Scripts (`scripts/archive/`)

These are older versions kept for reference but not actively used.

#### Weight Transfer Archive
- `transfer_weights_v2.py` - Older weight transfer version
- `transfer_weights_improved.py` - Intermediate improvement
- `transfer_vampnet_weights.py` - Original transfer attempt
- `load_pretrained_weights.py` - Early weight loading
- `extract_vampnet_weights.py` - Weight extraction utilities

#### Export Archive
- `export_vampnet_transformer.py` - Original transformer export
- `export_c2f_transformer.py` - Original C2F export
- `export_c2f_transformer_simple.py` - Simplified C2F attempt
- `export_pretrained_transformer.py` - Pretrained model export
- `export_torchscript_transformer.py` - TorchScript alternative

#### Analysis Archive
- `analyze_vampnet_structure.py` - VampNet architecture analysis
- `analyze_vampnet_ffn.py` - Feed-forward network analysis
- `analyze_weight_mapping.py` - Weight mapping analysis

#### Debug Archive
- `debug_ffn_weights.py` - FFN weight debugging
- `diagnose_transformer_shapes.py` - Shape mismatch diagnosis
- `fix_transformer_export.py` - Transformer export fixes
- `fix_ffn_weight_transfer.py` - FFN transfer fixes
- `fix_onnx_types.py` - Type inference fixes

## Notebooks (`notebooks/`)

### Main Notebooks
- **`vampnet_onnx_export.ipynb`** - Step-by-step export tutorial
- **`vampnet_onnx_audio_comparison.ipynb`** - Audio generation comparison
- **`vampnet_onnx_comparison_prepadded.ipynb`** ⭐ - Comparison using pre-padded encoder
- **`vampnet_onnx_optimization.ipynb`** - Model optimization and quantization
- **`vampnet_onnx_pipeline_test.ipynb`** - Pipeline testing and validation
- **`vampnet_complete_comparison_demo.ipynb`** - Complete comparison demo

### Development Notebooks
- **`complete_weight_transfer.ipynb`** - Interactive weight transfer experiments
- **`test_vampnet_import.ipynb`** - Testing VampNet imports
- **`vampnet_custom_ops_export.ipynb`** - Custom operator development
- **`vampnet_torchscript_export.ipynb`** - TorchScript export experiments

## Test Files (`tests/`)

- **`test_audio_processor.py`** - Unit tests for audio preprocessing

## Root Directory Test Scripts

These should probably be in `scripts/` but are currently in root:

- **`test_codec_comparison.py`** - Compares codec outputs
- **`test_codec_direct.py`** - Direct codec testing
- **`test_codec_export.py`** - Codec export validation
- **`test_onnx_export_codec.py`** - ONNX codec export tests
- **`test_onnx_inference.py`** - ONNX inference testing
- **`test_preprocessing_comparison.py`** - Preprocessing validation
- **`test_simplified_pipeline.py`** - Simplified pipeline testing
- **`test_prepadded_encoder.py`** - Pre-padded encoder validation
- **`test_vampnet_generation_strategy.py`** - VampNet generation analysis
- **`analyze_onnx_generation_issue.py`** - Generation issue diagnosis
- **`check_model_weights.py`** - Weight comparison tool
- **`test_weighted_vs_unweighted.py`** - Model comparison

## Documentation (`docs/`)

- **`onnx_export_summary.md`** - Export process summary
- **`quantization_analysis_summary.md`** - Quantization results
- **`onnx_generation_mismatch_analysis.md`** ⭐ - Analysis of generation issues
- **`weight_transfer_technical_notes.md`** ⭐ - Technical weight transfer details
- **`file_structure_guide.md`** - This file

## Configuration Files

- **`pyproject.toml`** - Python package configuration
- **`CLAUDE.md`** - Instructions for Claude AI assistance
- **`README.md`** - Main project documentation
- **`CONTRIBUTING.md`** - Contribution guidelines
- **`KNOWN_ISSUES.md`** - List of known issues

## Model Directories

- **`models/`** - VampNet pretrained models (not in git)
- **`scripts/models/`** - Exported ONNX models
- **`scripts/onnx_models_fixed/`** - Fixed/weighted ONNX models
- **`outputs/`** - Generated audio outputs
- **`assets/`** - Example audio files

## Generated Files (Not in Git)

- **`*.onnx`** - Exported ONNX models
- **`*.pth`** - PyTorch checkpoints
- **`*.wav`** - Audio outputs
- **`*.png`** - Visualization outputs

## Key File Relationships

### Export Pipeline
1. `export_vampnet_transformer_v2.py` → Creates transformer structure
2. `transfer_weights_vampnet_v2.py` → Transfers weights
3. `complete_weight_transfer.py` → Attempts to complete transfer
4. `vampnet_full_pipeline_fixed.py` → Runs complete pipeline

### Testing Pipeline
1. `test_codec_comparison.py` → Validates encoder tokens
2. `test_transformer_inference.py` → Tests generation
3. `compare_onnx_vs_original.py` → Compares outputs
4. `vampnet_onnx_comparison_fixed.py` → Full comparison

### Custom Operators
All custom ops in `scripts/custom_ops/` are imported by export scripts to handle ONNX limitations.

## Usage Patterns

### For Export
```bash
python scripts/export_vampnet_transformer_v2.py
python scripts/transfer_weights_vampnet_v2.py
```

### For Testing
```bash
python scripts/test_complete_model.py
python scripts/vampnet_onnx_comparison_fixed.py
```

### For Analysis
```bash
python scripts/analyze_onnx_model.py
python scripts/check_model_weights.py
```

## Notes

- Files with "v2" suffix are newer versions
- Files with "fixed" suffix have bug fixes
- Files with "proper" suffix use correct implementation
- Archive folder contains older attempts for reference
- Test files in root should be moved to scripts/