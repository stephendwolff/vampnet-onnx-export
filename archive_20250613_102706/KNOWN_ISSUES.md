# Known Issues

## 1. ONNX Generation Quality Mismatch

**Issue**: ONNX transformer models produce significantly different audio outputs compared to VampNet, despite the encoder producing nearly identical tokens (97.9% match).

**Status**: 🔴 Major Issue - Affects generation quality

**Root Cause**: 
- Incomplete weight transfer from VampNet to ONNX models
- Only transformer layer weights are transferred (121 out of 374 parameters)
- Embeddings and output classifiers are not properly transferred

**Details**:
- VampNet uses codec embeddings directly; ONNX uses standard embedding tables with random initialization
- Output dimension mismatch: VampNet uses 4096 classes, ONNX uses 1025
- VampNet has special mask token handling that isn't replicated in ONNX

**Workarounds**:
1. Use hybrid approach: VampNet transformer with ONNX codec
2. Accept quality difference for deployment scenarios
3. Wait for complete weight transfer implementation

**Related Files**:
- `docs/onnx_generation_mismatch_analysis.md`
- `docs/weight_transfer_technical_notes.md`

---

## 2. Fixed Sequence Length Limitation

**Issue**: ONNX transformer models require exactly 100 tokens and cannot handle variable-length sequences.

**Status**: 🟡 Moderate Issue - Has workaround

**Root Cause**: ONNX tracing captures fixed dimensions during export

**Workaround**: Audio is automatically chunked into 100-token segments by the pipeline

**Impact**: May affect generation quality at chunk boundaries

---

## 3. Mask Shape Errors in Weight Transfer

**Issue**: Running `complete_weight_transfer.py` produces mask shape mismatch errors.

**Status**: 🟢 Fixed - But export incomplete

**Error Message**:
```
The shape of the mask [1, 100] at index 1 does not match the shape of the indexed tensor [1, 4, 100] at index 1
```

**Solution**: Fixed in code - mask shape changed from `[batch, seq_len]` to `[batch, n_codebooks, seq_len]`

---

## 4. Pre-Padded Encoder Requirement

**Issue**: ONNX encoder requires audio to be pre-padded to multiples of 768 samples.

**Status**: 🟢 Resolved - With limitations

**Details**:
- Original ONNX encoder produced only 0.21% token match
- Pre-padded encoder achieves 97.9% match but requires fixed input sizes
- Different audio lengths need different encoder exports

**Workaround**: Use `pad_audio_for_encoder()` function before encoding

---

## 5. Weight Statistics Mismatch

**Issue**: ONNX model weights have different statistics than VampNet weights.

**Status**: 🔴 Unresolved

**Evidence**:
```
ONNX embeddings: mean≈0.0000, std=0.0200 (random initialization)
VampNet embeddings: mean=-0.0190, std=0.3573 (trained weights)
```

**Impact**: Major contributor to generation quality issues

---

## 6. Missing Sampling/Temperature Support

**Issue**: ONNX models use argmax for token generation instead of proper sampling.

**Status**: 🟡 Limitation

**Impact**: Less diverse/creative outputs compared to VampNet

**Workaround**: None currently - requires custom ONNX operators

---

## Report New Issues

Please report new issues with:
1. Error message or unexpected behavior
2. Steps to reproduce
3. Expected vs actual output
4. Environment details (OS, Python version, package versions)