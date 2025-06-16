"""
VampNet ONNX Export Module

This module provides utilities for exporting VampNet components to ONNX format.
"""

from .audio_processor import AudioProcessor
from .codec_wrapper import CodecEncoder, CodecDecoder
from .mask_generator import MaskGenerator
from .mask_generator_onnx import ONNXMaskGenerator, FlexibleONNXMaskGenerator
from .mask_generator_proper import build_mask_proper, ProperMaskGenerator
from .transformer_wrapper import TransformerWrapper
from .models import CoarseTransformer, C2FTransformer, VampNetTransformer
from .weight_transfer import WeightTransferManager, complete_weight_transfer
from .embeddings import (
    CodecEmbeddingExtractor,
    extract_and_convert_embeddings,
    load_embeddings_into_model
)
from .exporters import (
    export_audio_processor,
    export_codec_encoder,
    export_codec_decoder,
    export_mask_generator,
    export_transformer,
    export_complete_transformer,
    export_pretrained_encoder,
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
from .sampling import (
    temperature_sample,
    top_p_sample,
    sample_from_logits,
    batch_sample_from_logits,
    iterative_sample
)
from .interface import Interface

__all__ = [
    # Core components
    'AudioProcessor',
    'CodecEncoder',
    'CodecDecoder',
    'MaskGenerator',
    'ONNXMaskGenerator',
    'FlexibleONNXMaskGenerator',
    'build_mask_proper',
    'ProperMaskGenerator',
    'TransformerWrapper',
    
    # Models
    'CoarseTransformer',
    'C2FTransformer',
    'VampNetTransformer',
    
    # Weight transfer
    'WeightTransferManager',
    'complete_weight_transfer',
    
    # Embeddings
    'CodecEmbeddingExtractor',
    'extract_and_convert_embeddings',
    'load_embeddings_into_model',
    
    # Export functions
    'export_audio_processor',
    'export_codec_encoder',
    'export_codec_decoder',
    'export_mask_generator',
    'export_transformer',
    'export_complete_transformer',
    'export_pretrained_encoder',
    'export_all_components',
    
    # Validation
    'validate_onnx_model',
    'compare_outputs',
    'analyze_model_size',
    'validate_vampnet_component',
    'create_onnx_session',
    'benchmark_model',
    
    # Pipeline
    'VampNetONNXPipeline',
    
    # Sampling
    'temperature_sample',
    'top_p_sample',
    'sample_from_logits',
    'batch_sample_from_logits',
    'iterative_sample',
    
    # Interface
    'Interface'
]