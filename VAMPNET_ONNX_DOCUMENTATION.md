# VampNet ONNX Export Documentation

## Project Overview

This project aimed to export VampNet's masked acoustic token modeling system to ONNX format for cross-platform deployment. VampNet is a parallel iterative music generation model that uses masked token prediction.

## Current Status

### What Works ✅

1. **Individual Component Exports**
   - **Encoder**: `scripts/models/vampnet_encoder_prepadded.onnx` (97.9% token accuracy)
   - **Coarse Transformer**: `vampnet_transformer_v11.onnx` (100% correlation with VampNet)
   - **Decoder**: `scripts/models/vampnet_codec_decoder.onnx` (functional)
   - **C2F Transformer**: `vampnet_c2f_transformer_v15.onnx` (exported but has issues)

2. **Pipeline Comparison Steps**
   - `pipeline_comparison/01-05_*.py` files demonstrate each component works individually
   - Step-by-step comparison shows high correlation between VampNet and ONNX outputs
   - Weight transfer from VampNet checkpoints successful

3. **Supporting Infrastructure**
   - `scripts/iterative_generation.py`: Implements VampNet's iterative masked generation
   - `scripts/unified_vamp_pipeline.py`: Attempted end-to-end pipeline
   - Codec embedding extraction and transfer working

### What Doesn't Work ❌

1. **End-to-End Audio Generation**
   - While individual components work, the complete pipeline produces poor quality audio
   - Integration issues between components not fully resolved

2. **C2F Model**
   - Produces NaN values for certain inputs (seed-dependent)
   - This appears to be an issue with the original VampNet C2F model, not the export

3. **Variable Length Sequences**
   - ONNX models are fixed to 100 token sequences
   - Cannot handle arbitrary length audio

## Technical Architecture

### VampNet Pipeline
```
1. Audio Input
2. Encode → Tokens (14 codebooks)
3. Create Mask (typically mask first 3 codebooks)
4. Coarse Generation (iterative, 4 codebooks)
5. Coarse-to-Fine (add codebooks 5-14)
6. Decode → Audio Output
```

### Key Components

#### 1. Audio Encoder
- Converts audio to discrete tokens
- 14 codebooks, 1024 vocabulary size each
- Pre-padding solution achieves 97.9% accuracy

#### 2. Coarse Transformer
- 20 layers, all using MultiHeadRelativeAttention
- Expects latents as input (not raw codes)
- Outputs logits for masked positions
- Uses iterative refinement (typically 12 steps)

#### 3. C2F Transformer
- 16 layers
- Takes all 14 codebooks, conditions on first 4
- Outputs predictions for codebooks 5-14
- Single forward pass (not iterative like coarse)

#### 4. Decoder
- Converts tokens back to audio
- Expects 14 codebooks (pads if given fewer)

## Key Discoveries

1. **VampNet expects latents, not codes**: The transformers take latents from `codec.quantizer.from_codes()`
2. **All layers use MultiHeadRelativeAttention**: Not just layer 0 as initially thought
3. **FFN uses 4x expansion**: 1280 → 5120 → GatedGELU (splits to 2560) → 1280
4. **Iterative generation is critical**: Single forward pass doesn't work
5. **Weight normalization must be removed**: Before weight transfer

## File Structure

```
/
├── models/vampnet/          # Original VampNet checkpoints
├── scripts/
│   ├── models/             # Exported ONNX models
│   ├── export_*.py         # Export scripts (many versions)
│   ├── iterative_generation.py  # Iterative generation logic
│   └── unified_vamp_pipeline.py # Attempted unified pipeline
├── pipeline_comparison/     # Step-by-step comparison scripts
├── notebooks/              # Various demo notebooks
└── tests/                  # Test files
```

## Lessons Learned

### What Went Wrong

1. **Complexity Underestimated**: VampNet's iterative generation with dynamic masking is complex to replicate
2. **Integration Issues**: Individual components work but integration failed
3. **Debugging Difficulty**: Hard to debug why ONNX produces poor audio when components seem correct
4. **Version Control**: Too many experimental versions made it hard to track what worked

### Technical Challenges

1. **Dynamic Control Flow**: ONNX doesn't handle iterative generation naturally
2. **Latent vs Code Confusion**: Initial attempts used codes instead of latents
3. **Mask Semantics**: VampNet's masking logic is subtle and easy to get wrong
4. **Numerical Precision**: Small differences compound over iterations

### What Should Have Been Done Differently

1. **Start Simple**: Should have started with single forward pass before iterative
2. **Better Testing**: Need audio quality metrics, not just tensor comparisons
3. **Version Control**: Should have saved working versions at each milestone
4. **Follow VampNet Exactly**: Too much improvisation instead of exact replication

## Recommendations for Future Work

1. **Debug the Integration**: 
   - Start with working pipeline_comparison scripts
   - Add one component at a time
   - Test audio quality at each step

2. **Simplify First**:
   - Get single forward pass working
   - Then add iterative generation
   - Finally add C2F

3. **Better Testing**:
   - Use perceptual audio metrics
   - Test with multiple seeds
   - Compare intermediate activations

4. **Alternative Approaches**:
   - Consider TorchScript instead of ONNX
   - Try simpler masking strategies first
   - Export with dynamic axes support

## Code Examples

### Correct VampNet Usage
```python
# This is how VampNet actually works
z = interface.encode(audio_signal)
mask = interface.build_mask(z, audio_signal)
z_generated = interface.vamp(z, mask=mask)
audio_out = interface.decode(z_generated)
```

### ONNX Should Match
```python
# ONNX should follow the same interface
z = onnx_encode(audio)
mask = create_mask(z)  # Same mask as VampNet
z_generated = onnx_vamp(z, mask)  # Iterative generation
audio_out = onnx_decode(z_generated)
```

## Conclusion

While significant progress was made in exporting individual VampNet components to ONNX, the complete end-to-end pipeline remains non-functional. The project demonstrates both the potential and challenges of exporting complex iterative models to ONNX.

The individual components work correctly, suggesting the issue lies in the integration or in subtle differences in the iterative generation process. Future work should focus on debugging the integration using the working pipeline comparison scripts as a foundation.

## References

- Original VampNet paper and code
- Pipeline comparison scripts in `pipeline_comparison/`
- Export scripts showing various attempts
- `VAMPNET_ONNX_FINAL_STATUS.md` for previous summary