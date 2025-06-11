# Documentation Overview

This directory contains detailed documentation for the VampNet ONNX Export project.

## Core Documentation

### 📋 [File Structure Guide](file_structure_guide.md)
Complete guide to all files in the project, explaining what each file does and how they relate to each other.

### 🚀 [Script Usage Guide](script_usage_guide.md)
How to use the key scripts, including export, weight transfer, and testing scripts with examples.

### 💾 [Model Files Overview](model_files_overview.md)
Details about all model files, their sizes, purposes, and relationships.

## Technical Analysis

### 🔍 [ONNX Generation Mismatch Analysis](onnx_generation_mismatch_analysis.md)
**Important**: Explains why ONNX models produce different outputs than VampNet, including root causes and recommendations.

### ⚙️ [Weight Transfer Technical Notes](weight_transfer_technical_notes.md)
Technical details about the weight transfer process, architecture differences, and implementation challenges.

### 📊 [Quantization Analysis Summary](quantization_analysis_summary.md)
Results from model quantization experiments, including size reduction and performance impact.

### 📝 [ONNX Export Summary](onnx_export_summary.md)
Overview of the export process and components.

## Quick Reference

### Most Important Scripts
1. **Export**: `export_vampnet_transformer_v2.py`
2. **Weights**: `transfer_weights_vampnet_v2.py`
3. **Test**: `vampnet_onnx_comparison_fixed.py`
4. **Pipeline**: `vampnet_full_pipeline_fixed.py`

### Key Findings
- ✅ Encoder achieves 97.9% token match with VampNet
- ⚠️ Transformer has incomplete weight transfer
- ❌ Generation quality differs due to architectural mismatches

### Model Locations
- VampNet models: `models/vampnet/*.pth`
- ONNX models: `scripts/models/*.onnx`
- Weighted models: `scripts/onnx_models_fixed/*_weighted.onnx`

## Navigation

- 🏠 Back to [Main README](../README.md)
- 🐛 View [Known Issues](../KNOWN_ISSUES.md)
- 🤝 See [Contributing Guidelines](../CONTRIBUTING.md)