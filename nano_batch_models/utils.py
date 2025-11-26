"""
Utility functions for model loading and inference.

This module provides helper functions for working with Mistral models,
including weight loading, tokenization, and generation utilities.
"""

from typing import Optional, Dict, Any
import torch
from pathlib import Path

from .config import MistralConfig
from .model import MistralForCausalLM


def load_mistral_model(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> MistralForCausalLM:
    """
    Load a Mistral model from HuggingFace or local path.
    
    Args:
        model_name_or_path: HuggingFace model ID or local path
                           (e.g., "mistralai/Mistral-7B-v0.1")
        device: Device to load model on ("cuda", "cpu", etc.)
        dtype: Data type for model weights (e.g., torch.bfloat16, torch.float16)
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        Loaded MistralForCausalLM model
        
    Example:
        >>> model = load_mistral_model(
        ...     "mistralai/Mistral-7B-v0.1",
        ...     device="cuda",
        ...     dtype=torch.bfloat16
        ... )
    """
    # Set default dtype if not provided
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Load model
    model = MistralForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        **kwargs
    )
    
    # Move to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode by default
    
    return model


def get_model_memory_footprint(model: MistralForCausalLM) -> Dict[str, Any]:
    """
    Calculate the memory footprint of a Mistral model.
    
    Args:
        model: MistralForCausalLM instance
        
    Returns:
        Dictionary with memory statistics:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - size_mb: Approximate size in megabytes
            - size_gb: Approximate size in gigabytes
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size (assumes float32, adjust if using different dtype)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_bytes": total_size,
        "size_mb": total_size / (1024 ** 2),
        "size_gb": total_size / (1024 ** 3),
    }


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


@torch.no_grad()
def generate_tokens(
    model: MistralForCausalLM,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> torch.LongTensor:
    """
    Simple greedy/sampling generation (not optimized for production use).
    
    This is a basic implementation for testing. For production, use the
    continuous batching engine with PagedAttention.
    
    Args:
        model: MistralForCausalLM instance
        input_ids: Input token IDs [batch_size, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, < 1.0 = sharper, > 1.0 = flatter)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        
    Returns:
        Generated token IDs [batch_size, seq_len + max_new_tokens]
    """
    device = input_ids.device
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # Forward pass
        if past_key_values is None:
            # First iteration: process full prompt
            logits, past_key_values = model(input_ids)
        else:
            # Subsequent iterations: only process last token
            logits, past_key_values = model(
                input_ids[:, -1:],
                past_key_values=past_key_values
            )
        
        # Get logits for last token
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float("-inf")
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float("-inf")
        
        # Sample next token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return input_ids
