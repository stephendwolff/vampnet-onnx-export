"""
VampNet ONNX Export Module

This module provides utilities for exporting VampNet components to ONNX format.
"""

from .audio_processor import AudioProcessor
from .codec_wrapper import CodecEncoder, CodecDecoder
from .mask_generator import MaskGenerator
from .mask_generator_onnx import ONNXMaskGenerator, FlexibleONNXMaskGenerator
from .transformer_wrapper import TransformerWrapper
from .exporters import (
    export_audio_processor,
    export_codec_encoder,
    export_codec_decoder,
    export_mask_generator,
    export_transformer,
    export_all_components
)
from .validation import (
    validate_onnx_model, 
    compare_outputs, 
    analyze_model_size,
    validate_vampnet_component,
    create_onnx_session,
    benchmark_model
)
from .pipeline import VampNetONNXPipeline

__all__ = [
    'AudioProcessor',
    'CodecEncoder',
    'CodecDecoder',
    'MaskGenerator',
    'ONNXMaskGenerator',
    'FlexibleONNXMaskGenerator',
    'TransformerWrapper',
    'export_audio_processor',
    'export_codec_encoder',
    'export_codec_decoder',
    'export_mask_generator',
    'export_transformer',
    'export_all_components',
    'validate_onnx_model',
    'compare_outputs',
    'analyze_model_size',
    'validate_vampnet_component',
    'create_onnx_session',
    'benchmark_model',
    'VampNetONNXPipeline'
]