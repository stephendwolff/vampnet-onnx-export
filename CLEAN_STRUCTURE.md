# Clean Repository Structure

After running `./cleanup_repository.sh`, your repository will have this minimal structure:

```
vampnet-onnx-export-cleanup/
│
├── vampnet_onnx/                    # Main package
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── mask_generator_onnx.py
│   ├── pipeline.py
│   ├── transformer_wrapper.py
│   └── vampnet_codec.py
│
├── models/                          # Pretrained VampNet models
│   └── vampnet/
│       ├── codec.pth
│       ├── coarse.pth
│       ├── c2f.pth
│       └── wavebeat.pth
│
├── onnx_models_fixed/               # Your ONNX exports
│   ├── coarse_complete_v3.onnx     # Complete weight transfer
│   └── c2f_complete_v3.onnx        # Complete weight transfer
│
├── scripts/                         # Minimal essential scripts
│   ├── models/                      # ONNX codec models
│   │   ├── vampnet_encoder_prepadded.onnx
│   │   └── vampnet_codec_decoder.onnx
│   └── (only essential scripts)
│
├── notebooks/                       # Only working notebook
│   └── vampnet_onnx_comparison_prepadded.ipynb
│
├── assets/                          # Test audio files
│   └── stargazing.wav
│
├── docs/                            # Documentation
│   └── weight_transfer_technical_notes.md
│
├── venv/                            # Virtual environment
│
├── .git/                            # Version control
├── .gitignore
├── .python-version
├── pyproject.toml                   # Package definition
├── README.md                        # Main documentation
├── CLAUDE.md                        # AI assistance docs
└── generate_audio_simple.py         # Simple usage example
```

## What Gets Archived

Everything else goes into `archive_YYYYMMDD_HHMMSS/`:
- 37+ test scripts from root
- All experimental scripts from scripts/
- Build artifacts
- Output files
- Old notebooks
- Analysis documents

## Essential Scripts to Keep/Create

1. **generate_audio_simple.py** - Simple audio generation example
2. **export_to_onnx.py** - Export VampNet models to ONNX (to be created)
3. **compare_models.py** - Compare VampNet vs ONNX (to be created)

## Benefits

- **Clear structure**: Easy to understand what does what
- **No confusion**: No multiple versions of the same functionality
- **Easy to use**: Simple scripts that show how to use the models
- **Documented**: Clear README and examples
- **Reproducible**: Can regenerate outputs anytime