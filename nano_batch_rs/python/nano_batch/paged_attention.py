"""
PagedAttention implementation with KV Cache management.

This module provides the core PagedAttention primitives for efficient LLM inference:
- paged_attention_fwd: Pure PyTorch PagedAttention kernel
- KVCache: Physical block management for key/value tensors

Note: Pure PyTorch implementation (no Triton) for maximum compatibility.
"""

import torch
import math
from typing import Dict, List, Tuple, Optional


def paged_attention_fwd(
    query: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: torch.Tensor,  # [batch_size, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [batch_size]
    block_size: int,
    max_context_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute PagedAttention forward pass using pure PyTorch.
    
    This implementation is less optimized than a Triton kernel but works on all platforms.
    
    Args:
        query: Query tensor [batch_size, num_heads, seq_len, head_dim]
        key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: Block table mapping [batch_size, max_num_blocks_per_seq]
        context_lens: Actual context length per sequence [batch_size]
        block_size: Number of tokens per block
        max_context_len: Maximum context length (optional)
        
    Returns:
        Attention output [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    
    # Validate inputs
    assert key_cache.shape == value_cache.shape, "Key and value cache must have same shape"
    assert key_cache.shape[1] == block_size, f"Key cache block size {key_cache.shape[1]} != {block_size}"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads (GQA)"
    
    # For decode phase, seq_len should be 1
    assert seq_len == 1, "This implementation currently only supports decode phase (seq_len=1)"
    
    # Allocate output
    output = torch.zeros_like(query)
    
    # Process each sequence in the batch
    for batch_idx in range(batch_size):
        context_len = context_lens[batch_idx].item()
        num_blocks = (context_len + block_size - 1) // block_size
        
        # Gather keys and values for this sequence
        # We'll reconstruct the full KV sequence from paged blocks
        seq_keys = []
        seq_values = []
        
        for block_idx in range(num_blocks):
            physical_block_id = block_table[batch_idx, block_idx].item()
            
            # Determine how many tokens in this block are valid
            block_start_pos = block_idx * block_size
            block_end_pos = min(block_start_pos + block_size, context_len)
            num_valid_tokens = block_end_pos - block_start_pos
            
            # Extract keys and values from this block
            # key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_keys = key_cache[physical_block_id, :num_valid_tokens]  # [num_valid, num_kv_heads, head_dim]
            block_values = value_cache[physical_block_id, :num_valid_tokens]
            
            seq_keys.append(block_keys)
            seq_values.append(block_values)
        
        # Concatenate all blocks to form full sequence
        full_keys = torch.cat(seq_keys, dim=0)  # [context_len, num_kv_heads, head_dim]
        full_values = torch.cat(seq_values, dim=0)  # [context_len, num_kv_heads, head_dim]
        
        # Reshape for attention: [num_kv_heads, context_len, head_dim]
        full_keys = full_keys.transpose(0, 1)
        full_values = full_values.transpose(0, 1)
        
        # Repeat KV heads for Grouped Query Attention
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            # [num_kv_heads, context_len, head_dim] -> [num_heads, context_len, head_dim]
            full_keys = full_keys.repeat_interleave(num_groups, dim=0)
            full_values = full_values.repeat_interleave(num_groups, dim=0)
        
        # Get query for this sequence: [num_heads, 1, head_dim]
        seq_query = query[batch_idx]
        
        # Compute attention scores: [num_heads, 1, context_len]
        scores = torch.matmul(seq_query, full_keys.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)
        
        # Apply causal mask (all past tokens are visible in decode)
        # For decode, we attend to all previous tokens
        # No masking needed since we're only generating 1 token
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply attention to values: [num_heads, 1, head_dim]
        seq_output = torch.matmul(attn_weights, full_values)
        
        # Store in output
        output[batch_idx] = seq_output
    
    return output


class KVCache:
    """
    Manages physical KV cache blocks for PagedAttention.
    
    The cache is organized as:
    - K: [num_blocks, block_size, num_kv_heads, head_dim]
    - V: [num_blocks, block_size, num_kv_heads, head_dim]
    
    Physical blocks are allocated by the Rust scheduler and this class
    provides the GPU tensors for storing the actual key/value data.
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize KV cache.
        
        Args:
            num_blocks: Total number of physical blocks
            block_size: Number of tokens per block
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension of each head
            dtype: Data type for cache
            device: Device to allocate cache on
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Allocate cache tensors
        cache_shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    
    def write_kv(
        self,
        keys: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        values: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        slot_mappings: torch.Tensor,  # [num_tokens] - physical slot indices
    ) -> None:
        """
        Write keys and values to the cache at specified physical slots.
        
        Args:
            keys: Key tensors to write
            values: Value tensors to write
            slot_mappings: Physical slot indices (block_id * block_size + offset)
        """
        num_tokens = keys.shape[0]
        
        # Convert slot indices to block IDs and offsets
        block_ids = slot_mappings // self.block_size
        block_offsets = slot_mappings % self.block_size
        
        # Write to cache using advanced indexing
        # This is efficient as it's a single scatter operation
        for i in range(num_tokens):
            block_id = block_ids[i].item()
            offset = block_offsets[i].item()
            self.key_cache[block_id, offset] = keys[i]
            self.value_cache[block_id, offset] = values[i]
    
    def get_kv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the full key and value cache tensors.
        
        Returns:
            Tuple of (key_cache, value_cache)
        """
        return self.key_cache, self.value_cache
    
    def reset(self) -> None:
        """Reset the cache to zeros (useful for benchmarking/testing)."""
        self.key_cache.zero_()
        self.value_cache.zero_()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of the KV cache.
        
        Returns:
            Dictionary with memory statistics in bytes and GB
        """
        key_size = self.key_cache.numel() * self.key_cache.element_size()
        value_size = self.value_cache.numel() * self.value_cache.element_size()
        total_size = key_size + value_size
        
        return {
            "key_cache_bytes": key_size,
            "value_cache_bytes": value_size,
            "total_bytes": total_size,
            "total_gb": total_size / (1024 ** 3),
        }


__all__ = ["paged_attention_fwd", "KVCache"]