"""
Mistral 7B model implementation with support for PagedAttention.

This module provides a PyTorch implementation of the Mistral model architecture,
designed to work with the nano_batch continuous batching engine.
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MistralConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # q, k: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    """Mistral MLP (SwiGLU activation)."""
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralAttention(nn.Module):
    """
    Multi-headed attention with Grouped Query Attention (GQA).
    
    NOTE: This is a standard implementation. PagedAttention will be integrated
    in a separate module/kernel.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads for Grouped Query Attention.
        
        hidden_states: [batch, num_key_value_heads, slen, head_dim]
        Returns: [batch, num_attention_heads, slen, head_dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key/value for autoregressive generation
            
        Returns:
            Tuple of (output, updated_key_value_cache)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Concatenate with past key/values if provided
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states)
        
        # Repeat k/v heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class MistralDecoderLayer(nn.Module):
    """Single Mistral decoder layer."""
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key/value
            
        Returns:
            Tuple of (output, updated_key_value_cache)
        """
        residual = hidden_states
        
        # Self-attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class MistralModel(nn.Module):
    """
    Mistral 7B base model.
    
    This is the core transformer without the language modeling head.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.padding_idx = None  # Mistral doesn't use padding
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached key/values for all layers
            
        Returns:
            Tuple of (hidden_states, updated_past_key_values)
        """
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through decoder layers
        new_past_key_values = () if past_key_values is not None else None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            
            if past_key_values is not None:
                new_past_key_values += (present_key_value,)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, new_past_key_values


class MistralForCausalLM(nn.Module):
    """
    Mistral model with a language modeling head.
    
    This is the complete model used for text generation.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached key/values
            
        Returns:
            Tuple of (logits, updated_past_key_values)
        """
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "MistralForCausalLM":
        """
        Load a pretrained Mistral model from HuggingFace.
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path
            **kwargs: Additional arguments passed to AutoModelForCausalLM
            
        Returns:
            MistralForCausalLM instance with loaded weights
        """
        try:
            from transformers import AutoModelForCausalLM
            
            # Load the HuggingFace model
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            
            # Get config
            config = MistralConfig.from_pretrained(model_name_or_path)
            
            # Create our model
            model = cls(config)
            
            # Copy weights (HuggingFace and our implementation have identical structure)
            model.load_state_dict(hf_model.state_dict())
            
            return model
            
        except ImportError:
            raise ImportError(
                "transformers package is required to load pretrained models. "
                "Install with: pip install transformers"
            )
