"""
nano_batch_models - Model Implementations for nano_batch

This package provides ready-to-use model implementations that work with
the nano_batch core scheduling engine.

Currently includes:
- Mistral 7B with full architecture (RoPE, GQA, SwiGLU)
- PagedAttention-optimized variants
- High-level inference engine
- Tokenizer utilities

Installation:
    Requires nano_batch_rs to be installed and built.
    
Usage:
    from nano_batch_models import MistralForCausalLM, PagedMistralForCausalLM
    from nano_batch_models import InferenceEngine, GenerationConfig
"""

from .models.mistral import (
    MistralConfig,
    MistralForCausalLM,
    MistralModel,
    MistralAttention,
    MistralDecoderLayer,
    MistralMLP,
    RMSNorm,
    RotaryEmbedding,
)

from .models.paged_mistral import (
    PagedMistralAttention,
    PagedMistralDecoderLayer,
)

from .inference_engine import (
    InferenceEngine,
    PagedMistralForCausalLM,
    GenerationConfig,
)

from .tokenizer import MistralTokenizer

__version__ = "0.1.0"

__all__ = [
    # Config
    "MistralConfig",
    
    # Standard Mistral
    "MistralForCausalLM",
    "MistralModel",
    "MistralAttention",
    "MistralDecoderLayer",
    "MistralMLP",
    "RMSNorm",
    "RotaryEmbedding",
    
    # Paged variants
    "PagedMistralAttention",
    "PagedMistralDecoderLayer",
    "PagedMistralForCausalLM",
    
    # Inference
    "InferenceEngine",
    "GenerationConfig",
    
    # Tokenizer
    "MistralTokenizer",
]
