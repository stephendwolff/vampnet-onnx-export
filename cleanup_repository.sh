#!/bin/bash

# Create archive directory with timestamp
ARCHIVE_DIR="archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "Archiving experimental files to $ARCHIVE_DIR..."

# Archive all test/analysis Python files from root
echo "Moving test files..."
mv test_*.py "$ARCHIVE_DIR/" 2>/dev/null
mv analyze_*.py "$ARCHIVE_DIR/" 2>/dev/null
mv check_*.py "$ARCHIVE_DIR/" 2>/dev/null
mv fix_*.py "$ARCHIVE_DIR/" 2>/dev/null
mv inspect_*.py "$ARCHIVE_DIR/" 2>/dev/null

# Archive analysis documents
echo "Moving analysis documents..."
mv vampnet_*.txt "$ARCHIVE_DIR/" 2>/dev/null
mv vampnet_*.md "$ARCHIVE_DIR/" 2>/dev/null
mv KNOWN_ISSUES.md "$ARCHIVE_DIR/" 2>/dev/null
mv CONTRIBUTING.md "$ARCHIVE_DIR/" 2>/dev/null

# Archive scripts folder but keep essentials
echo "Archiving scripts folder..."
cp -r scripts "$ARCHIVE_DIR/" 2>/dev/null

# Keep essential scripts
ESSENTIAL_SCRIPTS=(
    "transfer_weights_complete_v3.py"
    "export_complete_model_to_onnx.py"
    "export_vampnet_transformer_v2.py"
    "export_working_encoder.py"
    "vampnet_full_pipeline_fixed.py"
    "extract_codec_embeddings.py"
)

# Keep custom ops directory
ESSENTIAL_DIRS=(
    "custom_ops"
    "models"  # Contains ONNX codec models
)

# Create new clean scripts directory
mkdir -p scripts_clean
mkdir -p scripts_clean/models

# Copy essential scripts
for script in "${ESSENTIAL_SCRIPTS[@]}"; do
    if [ -f "scripts/$script" ]; then
        cp "scripts/$script" "scripts_clean/"
        echo "  Keeping: $script"
    fi
done

# Copy essential directories
for dir in "${ESSENTIAL_DIRS[@]}"; do
    if [ -d "scripts/$dir" ]; then
        cp -r "scripts/$dir" "scripts_clean/"
        echo "  Keeping: $dir/"
    fi
done

# Remove old scripts and rename clean version
rm -rf scripts
mv scripts_clean scripts

# Move outputs (can be regenerated)
echo "Moving outputs..."
mv outputs "$ARCHIVE_DIR/" 2>/dev/null

# Move build artifacts
echo "Moving build artifacts..."
mv build "$ARCHIVE_DIR/" 2>/dev/null
mv *.egg-info "$ARCHIVE_DIR/" 2>/dev/null

# Move test_audio if it duplicates assets
if [ -d "test_audio" ] && [ -d "assets" ]; then
    echo "Moving test_audio..."
    mv test_audio "$ARCHIVE_DIR/" 2>/dev/null
fi

# Archive old notebooks except the working prepadded one
echo "Archiving old notebooks..."
mkdir -p "$ARCHIVE_DIR/notebooks"
find notebooks -name "*.ipynb" -not -name "vampnet_onnx_comparison_prepadded.ipynb" -exec mv {} "$ARCHIVE_DIR/notebooks/" \; 2>/dev/null

# Create a minimal scripts directory with only essentials
echo "Creating minimal scripts directory..."
mkdir -p scripts

# Create a manifest of what was archived
echo "Creating archive manifest..."
cat > "$ARCHIVE_DIR/ARCHIVE_MANIFEST.txt" << EOF
Archive created on $(date)
========================

This archive contains experimental and test files from the VampNet ONNX export project.

Contents:
- Test scripts (test_*.py)
- Analysis scripts (analyze_*.py, check_*.py, etc.)
- Original scripts folder with all experimental versions
- Build artifacts
- Output files
- Analysis documents

The main repository has been cleaned to contain only:
- vampnet_onnx/ - main package
- models/ - pretrained models
- onnx_models_fixed/ - ONNX exports
- notebooks/vampnet_onnx_comparison_prepadded.ipynb - working notebook
- assets/ - test audio
- Core documentation (README.md, CLAUDE.md)
- Package files (pyproject.toml, .gitignore, etc.)
EOF

echo "
Cleanup complete! 

Archived to: $ARCHIVE_DIR/

The repository now contains only essential files.
You may want to create a few minimal scripts in the scripts/ directory:
1. export_models.py - for exporting models to ONNX
2. run_comparison.py - for running comparisons
3. generate_audio.py - for simple audio generation
"