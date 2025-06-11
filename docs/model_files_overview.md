# Model Files Overview

This document describes the various model files in the project, their purposes, and relationships.

## Model Directory Structure

```
vampnet-onnx-export-cleanup/
├── models/                          # VampNet pretrained models (not in git)
│   └── vampnet/
│       ├── codec.pth               # DAC codec weights
│       ├── coarse.pth              # Coarse transformer weights
│       ├── c2f.pth                 # Coarse-to-fine transformer weights
│       └── wavebeat.pth            # Wavebeat model weights
│
├── scripts/
│   ├── models/                      # Exported ONNX models
│   │   ├── vampnet_codec_encoder.onnx
│   │   ├── vampnet_codec_decoder.onnx
│   │   └── vampnet_encoder_prepadded.onnx
│   │
│   └── onnx_models_fixed/           # Weighted/fixed ONNX models
│       ├── coarse_transformer_v2.onnx
│       ├── coarse_transformer_v2_weighted.onnx
│       ├── c2f_transformer_v2.onnx
│       └── c2f_transformer_v2_weighted.onnx
│
└── outputs/                         # Generated outputs
    └── comparison_fixed/
        ├── 01_original.wav
        ├── 02_vampnet.wav
        └── 03_onnx.wav
```

## VampNet Pretrained Models (`models/vampnet/`)

### codec.pth
- **Size**: ~180 MB
- **Purpose**: DAC (Descript Audio Codec) weights
- **Contents**: Encoder and decoder for audio tokenization
- **Usage**: Converts audio ↔ discrete tokens

### coarse.pth
- **Size**: ~1.5 GB
- **Purpose**: Coarse transformer for initial token generation
- **Contents**: 20-layer transformer, generates 4 coarse codebooks
- **Architecture**: 1280 dim, 20 heads, GatedGELU activation

### c2f.pth
- **Size**: ~1.3 GB
- **Purpose**: Coarse-to-fine transformer
- **Contents**: Refines coarse tokens, adds 10 fine codebooks
- **Architecture**: Similar to coarse but different output dims

### wavebeat.pth
- **Size**: ~50 MB
- **Purpose**: Beat tracking and rhythm analysis
- **Usage**: Optional component for rhythm-aware generation

## ONNX Codec Models (`scripts/models/`)

### vampnet_codec_encoder.onnx
- **Size**: 91.4 MB
- **Purpose**: Encode audio to tokens
- **Input**: Audio tensor `[batch, channels, samples]`
- **Output**: Token tensor `[batch, 14, seq_len]`
- **Limitation**: Fixed output length due to ONNX tracing

### vampnet_codec_decoder.onnx
- **Size**: 90.8 MB
- **Purpose**: Decode tokens to audio
- **Input**: Token tensor `[batch, 14, seq_len]`
- **Output**: Audio tensor `[batch, channels, samples]`
- **Note**: Handles all 14 codebooks

### vampnet_encoder_prepadded.onnx
- **Size**: 91.4 MB
- **Purpose**: Fixed-size encoder with 97.9% token accuracy
- **Input**: Pre-padded audio (multiple of 768 samples)
- **Output**: Tokens matching VampNet's output
- **Key Feature**: Best token matching with original

## ONNX Transformer Models (`scripts/onnx_models_fixed/`)

### coarse_transformer_v2.onnx
- **Size**: 1.38 GB
- **Purpose**: Coarse token generation (unweighted)
- **Input**: 
  - `codes`: `[batch, 4, 100]`
  - `mask`: `[batch, 4, 100]`
- **Output**: `[batch, 4, 100]`
- **Issue**: Random weight initialization

### coarse_transformer_v2_weighted.onnx
- **Size**: 1.38 GB
- **Purpose**: Coarse generation with partial VampNet weights
- **Improvement**: 121 transformer parameters transferred
- **Limitation**: Missing embeddings and classifiers

### c2f_transformer_v2.onnx
- **Size**: 1.22 GB
- **Purpose**: Fine token generation (unweighted)
- **Input**: 
  - `codes`: `[batch, 14, 100]`
  - `mask`: `[batch, 14, 100]`
- **Output**: `[batch, 14, 100]`

### c2f_transformer_v2_weighted.onnx
- **Size**: 1.22 GB
- **Purpose**: Fine generation with partial weights
- **Feature**: Preserves coarse codes, generates fine codes

## Model Relationships

```
Audio Input
    ↓
vampnet_encoder_prepadded.onnx (Best accuracy)
    ↓
Tokens [batch, 14, seq_len]
    ↓
coarse_transformer_v2_weighted.onnx (First 4 codebooks)
    ↓
c2f_transformer_v2_weighted.onnx (All 14 codebooks)
    ↓
vampnet_codec_decoder.onnx
    ↓
Audio Output
```

## Model Quality Comparison

| Model | Token Accuracy | Generation Quality | Notes |
|-------|---------------|-------------------|--------|
| VampNet (PyTorch) | 100% | Excellent | Original implementation |
| vampnet_codec_encoder.onnx | 0.21% | N/A | Fixed size issue |
| vampnet_encoder_prepadded.onnx | 97.9% | N/A | Best ONNX encoder |
| coarse_transformer_v2.onnx | N/A | Poor | Random weights |
| coarse_transformer_v2_weighted.onnx | N/A | Better | Partial weights |

## Weight Transfer Status

### Successfully Transferred
- Transformer attention layers ✅
- Layer normalization (RMSNorm) ✅
- Feed-forward networks ✅
- FiLM conditioning ✅

### Not Transferred
- Embeddings ❌ (architectural mismatch)
- Output classifiers ❌ (dimension mismatch)
- Positional encodings ❌ (different approach)
- Special tokens ❌ (incompatible structure)

## File Size Analysis

### Optimization Results
- Original VampNet: ~3 GB total
- ONNX models: ~2.8 GB total
- Quantized models: ~1.8 GB total (35% reduction)

### Storage Requirements
- Minimum (codec only): ~180 MB
- Full pipeline: ~2.8 GB
- With VampNet models: ~5.8 GB

## Usage Examples

### Load ONNX Models
```python
import onnxruntime as ort

# Load encoder
encoder = ort.InferenceSession("scripts/models/vampnet_encoder_prepadded.onnx")

# Load transformer
transformer = ort.InferenceSession("scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx")

# Load decoder
decoder = ort.InferenceSession("scripts/models/vampnet_codec_decoder.onnx")
```

### Check Model Info
```python
# Get input/output shapes
for input in encoder.get_inputs():
    print(f"Input: {input.name}, shape: {input.shape}, type: {input.type}")

for output in encoder.get_outputs():
    print(f"Output: {output.name}, shape: {output.shape}, type: {output.type}")
```

## Recommendations

1. **For Best Token Accuracy**: Use `vampnet_encoder_prepadded.onnx`
2. **For Generation**: Use weighted transformer models
3. **For Quality**: Consider hybrid approach (PyTorch transformer + ONNX codec)
4. **For Deployment**: Quantize models for size/speed

## Known Issues

1. **Fixed Sequence Length**: All transformers require exactly 100 tokens
2. **Incomplete Weights**: Only transformer layers transferred
3. **Quality Gap**: ONNX generation differs from VampNet
4. **Size Constraints**: Large model files may be problematic for mobile

See [ONNX Generation Mismatch Analysis](onnx_generation_mismatch_analysis.md) for detailed analysis.