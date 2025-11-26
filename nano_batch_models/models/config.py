"""
Model configuration for Mistral 7B.

This module defines the configuration parameters for the Mistral architecture,
matching the official implementation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MistralConfig:
    """Configuration for Mistral 7B model."""
    
    # Model dimensions
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA (Grouped Query Attention)
    
    # Vocabulary
    vocab_size: int = 32000
    
    # Context length
    max_position_embeddings: int = 32768  # Mistral 7B v0.2+ supports 32k
    
    # RoPE (Rotary Position Embeddings)
    rope_theta: float = 10000.0
    
    # Attention
    attention_dropout: float = 0.0
    
    # Normalization
    rms_norm_eps: float = 1e-5
    
    # Activation
    hidden_act: str = "silu"
    
    # Sliding Window Attention (SWA)
    sliding_window: Optional[int] = 4096  # None to disable
    
    # Dtype
    torch_dtype: str = "bfloat16"
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "MistralConfig":
        """
        Load configuration from a pretrained model.
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path
            
        Returns:
            MistralConfig instance
        """
        try:
            from transformers import AutoConfig
            
            hf_config = AutoConfig.from_pretrained(model_name_or_path)
            
            return cls(
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
                vocab_size=hf_config.vocab_size,
                max_position_embeddings=hf_config.max_position_embeddings,
                rope_theta=getattr(hf_config, "rope_theta", 10000.0),
                attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
                rms_norm_eps=hf_config.rms_norm_eps,
                hidden_act=hf_config.hidden_act,
                sliding_window=getattr(hf_config, "sliding_window", None),
            )
        except ImportError:
            raise ImportError(
                "transformers package is required to load pretrained configs. "
                "Install with: pip install transformers"
            )
