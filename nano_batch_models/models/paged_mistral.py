"""
PagedAttention variant of Mistral model.

This module provides a Mistral attention layer that uses the custom
PagedAttention kernel for efficient inference with the continuous batching engine.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import MistralConfig
from .mistral import RotaryEmbedding, apply_rotary_pos_emb
from nano_batch import paged_attention_fwd


class PagedMistralAttention(nn.Module):
    """
    PagedAttention variant of MistralAttention.
    
    This uses the PagedAttention kernel for efficient KV cache access
    during the decode phase. Keeps the same interface as MistralAttention
    for easy drop-in replacement.
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
        
        # Projections (same as standard attention)
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
        hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        key_cache: Optional[torch.Tensor] = None,  # [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Optional[torch.Tensor] = None,  # [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: Optional[torch.Tensor] = None,  # [batch_size, max_blocks_per_seq]
        context_lens: Optional[torch.Tensor] = None,  # [batch_size]
        block_size: Optional[int] = None,
        slot_mappings: Optional[torch.Tensor] = None,  # [num_tokens] for writing new KV
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with PagedAttention or standard attention fallback.
        
        Args:
            hidden_states: Input hidden states
            key_cache: Physical key cache blocks
            value_cache: Physical value cache blocks
            block_table: Mapping from logical blocks to physical blocks
            context_lens: Context length for each sequence
            block_size: Number of tokens per block
            slot_mappings: Physical slot indices for writing new KV
            
        Returns:
            Tuple of (output, (new_keys, new_values))
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
        
        # Apply rotary embeddings to current tokens
        # For decode: seq_len=1, for prefill: seq_len>1
        kv_seq_len = context_lens.max().item() if context_lens is not None else seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # Slice for current sequence length (important for decode where seq_len=1 but kv_seq_len>1)
        cos = cos[-seq_len:, :]
        sin = sin[-seq_len:, :]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Return K, V for cache update (before using attention)
        # These will be written to the cache using slot_mappings
        new_keys = key_states.transpose(1, 2).contiguous()  # [batch, seq_len, num_kv_heads, head_dim]
        new_values = value_states.transpose(1, 2).contiguous()
        
        # Choose attention implementation based on sequence length
        if seq_len > 1:
            # PREFILL PHASE: Use standard PyTorch attention
            # Repeat k/v heads for GQA
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
            
            # Scaled dot-product attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
        else:
            # DECODE PHASE: Use PagedAttention kernel (seq_len == 1)
            assert key_cache is not None and value_cache is not None, "KV cache required for decode phase"
            assert block_table is not None and context_lens is not None and block_size is not None
            
            attn_output = paged_attention_fwd(
                query_states,
                key_cache,
                value_cache,
                block_table,
                context_lens,
                block_size,
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, (new_keys, new_values)


class PagedMistralDecoderLayer(nn.Module):
    """
    Decoder layer using PagedAttention.
    
    This is identical to MistralDecoderLayer except it uses PagedMistralAttention.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        from .mistral import MistralMLP, RMSNorm
        
        self.self_attn = PagedMistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        slot_mappings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with PagedAttention."""
        residual = hidden_states
        
        # Self-attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, (new_keys, new_values) = self.self_attn(
            hidden_states=hidden_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            context_lens=context_lens,
            block_size=block_size,
            slot_mappings=slot_mappings,
        )
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, (new_keys, new_values)
