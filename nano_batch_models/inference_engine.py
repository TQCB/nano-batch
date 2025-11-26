"""
Inference engine integrating Rust scheduler with Python model and PagedAttention.

This module provides the main InferenceEngine class that coordinates:
- Rust block allocator and scheduler
- PyTorch Mistral model with PagedAttention
- KV cache management
- Token sampling and generation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nano_batch import Engine as RustEngine, KVCache
from .models.config import MistralConfig
from .models.paged_mistral import PagedMistralDecoderLayer
from .models.mistral import MistralForCausalLM, RMSNorm
from .tokenizer import MistralTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stop_tokens: Optional[List[int]] = None


class PagedMistralForCausalLM(torch.nn.Module):
    """
    Mistral model adapted for PagedAttention inference.
    
    This replaces standard attention layers with PagedMistralAttention layers.
    """
    
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers with PagedAttention
        self.layers = torch.nn.ModuleList([
            PagedMistralDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        slot_mappings: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with PagedAttention.
        
        Returns:
            (logits, list of (new_keys, new_values) per layer)
        """
        hidden_states = self.embed_tokens(input_ids)
        
        new_kv_list = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, (new_keys, new_values) = layer(
                hidden_states,
                key_cache=key_caches[layer_idx],
                value_cache=value_caches[layer_idx],
                block_table=block_table,
                context_lens=context_lens,
                block_size=block_size,
                slot_mappings=slot_mappings,
            )
            new_kv_list.append((new_keys, new_values))
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, new_kv_list
    
    @classmethod
    def from_standard_mistral(cls, standard_model: MistralForCausalLM) -> "PagedMistralForCausalLM":
        """
        Convert a standard MistralForCausalLM to PagedMistralForCausalLM.
        
        Copies weights from standard model to paged model.
        """
        config = standard_model.config
        paged_model = cls(config)
        
        # Copy embedding weights
        paged_model.embed_tokens.weight.data = standard_model.model.embed_tokens.weight.data.clone()
        
        # Copy layer weights
        for paged_layer, standard_layer in zip(paged_model.layers, standard_model.model.layers):
            # Attention projections
            paged_layer.self_attn.q_proj.weight.data = standard_layer.self_attn.q_proj.weight.data.clone()
            paged_layer.self_attn.k_proj.weight.data = standard_layer.self_attn.k_proj.weight.data.clone()
            paged_layer.self_attn.v_proj.weight.data = standard_layer.self_attn.v_proj.weight.data.clone()
            paged_layer.self_attn.o_proj.weight.data = standard_layer.self_attn.o_proj.weight.data.clone()
            
            # MLP
            paged_layer.mlp.gate_proj.weight.data = standard_layer.mlp.gate_proj.weight.data.clone()
            paged_layer.mlp.up_proj.weight.data = standard_layer.mlp.up_proj.weight.data.clone()
            paged_layer.mlp.down_proj.weight.data = standard_layer.mlp.down_proj.weight.data.clone()
            
            # LayerNorms
            paged_layer.input_layernorm.weight.data = standard_layer.input_layernorm.weight.data.clone()
            paged_layer.post_attention_layernorm.weight.data = standard_layer.post_attention_layernorm.weight.data.clone()
        
        # Copy final norm and LM head
        paged_model.norm.weight.data = standard_model.model.norm.weight.data.clone()
        paged_model.lm_head.weight.data = standard_model.lm_head.weight.data.clone()
        
        return paged_model


class InferenceEngine:
    """
    Main inference engine coordinating Rust scheduler and Python model.
    
    This class integrates all components:
    - Rust block allocator/scheduler
    - PyTorch model with PagedAttention
    - KV cache management
    - Tokenization
    """
    
    def __init__(
        self,
        model: PagedMistralForCausalLM,
        tokenizer: MistralTokenizer,
        num_blocks: int,
        block_size: int,
        device: str = "cuda",
    ):
        """
        Initialize inference engine.
        
        Args:
            model: PagedMistralForCausalLM model
            tokenizer: Tokenizer for encoding/decoding
            num_blocks: Total number of KV cache blocks
            block_size: Tokens per block
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.config = model.config
        
        # Initialize Rust scheduler
        self.rust_engine = RustEngine(num_blocks, block_size)
        
        # Initialize KV caches (one per layer)
        self.kv_caches = [
            KVCache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        
        # Track active requests
        self.active_requests: Dict[str, Dict] = {}
        self.request_counter = 0
    
    def add_request(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Add a new generation request.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            
        Returns:
            Request ID
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Generate request ID
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        # Add to Rust scheduler
        self.rust_engine.add_request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=generation_config.max_tokens,
            stop_tokens=generation_config.stop_tokens or [],
        )
        
        # Track locally
        self.active_requests[request_id] = {
            "prompt": prompt,
            "prompt_token_ids": token_ids,
            "output_tokens": [],
            "config": generation_config,
        }
        
        return request_id
    
    @torch.no_grad()
    def step(self) -> Dict[str, str]:
        """
        Run one inference step.
        
        Returns:
            Dictionary of request_id -> newly generated token text
        """
        # Get scheduled requests from Rust
        scheduler_output = self.rust_engine.step()
        
        if not scheduler_output.scheduled_requests:
            return {}
        
        # Prepare inputs
        batch_size = len(scheduler_output.scheduled_requests)
        input_ids_list = []
        
        for req_id in scheduler_output.scheduled_requests:
            # Get all tokens for this request (prompt + generated)
            prompt_ids = self.active_requests[req_id]["prompt_token_ids"]
            output_ids = self.active_requests[req_id]["output_tokens"]
            all_tokens = prompt_ids + output_ids
            
            # Number of tokens to process in this step
            num_tokens = scheduler_output.num_tokens_per_request[req_id]
            
            # For prefill (num_tokens > 1), process all prompt tokens
            # For decode (num_tokens == 1), process the last token (most recently generated)
            if num_tokens > 1:
                # Prefill: use prompt tokens
                tokens_to_process = prompt_ids[:num_tokens]
            else:
                # Decode: use the last token in the sequence
                tokens_to_process = [all_tokens[-1]] if all_tokens else [0]
            
            input_ids_list.append(tokens_to_process)
        
        # Pad inputs to max_len
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        
        # Create mask for valid tokens [batch_size, max_len]
        valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        
        # Store lengths for sampling
        seq_lengths = []
        
        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            seq_lengths.append(seq_len)
            pad_len = max_len - seq_len
            padded_input_ids.append(ids + [0] * pad_len)
            valid_mask[i, :seq_len] = True
        
        # Create tensor [batch_size, max_len]
        input_ids = torch.tensor(padded_input_ids, device=self.device, dtype=torch.long)
        
        # Prepare block tables
        max_blocks = max(len(scheduler_output.block_tables[req_id]) for req_id in scheduler_output.scheduled_requests)
        block_table = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device=self.device)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        for i, req_id in enumerate(scheduler_output.scheduled_requests):
            blocks = scheduler_output.block_tables[req_id]
            block_table[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            # Calculate exact context length
            prompt_ids = self.active_requests[req_id]["prompt_token_ids"]
            output_ids = self.active_requests[req_id]["output_tokens"]
            context_lens[i] = len(prompt_ids) + len(output_ids)
        
        # Slot mappings for writing new KV
        slot_mappings = torch.tensor(scheduler_output.slot_mappings, dtype=torch.long, device=self.device)
        
        # Get KV caches
        key_caches = [cache.key_cache for cache in self.kv_caches]
        value_caches = [cache.value_cache for cache in self.kv_caches]
        
        # Forward pass
        logits, new_kv_list = self.model(
            input_ids,
            key_caches,
            value_caches,
            block_table,
            context_lens,
            self.block_size,
            slot_mappings,
        )
        
        # Write new KV to cache
        flat_mask = valid_mask.view(-1)
        
        for layer_idx, (new_keys, new_values) in enumerate(new_kv_list):
            # Reshape: [batch, seq_len, num_kv_heads, head_dim] -> [batch * seq_len, num_kv_heads, head_dim]
            batch, seq_len, num_kv_heads, head_dim = new_keys.shape
            
            flat_keys = new_keys.reshape(batch * seq_len, num_kv_heads, head_dim)
            flat_values = new_values.reshape(batch * seq_len, num_kv_heads, head_dim)
            
            # Filter out padding tokens using mask
            valid_keys = flat_keys[flat_mask]
            valid_values = flat_values[flat_mask]
            
            self.kv_caches[layer_idx].write_kv(valid_keys, valid_values, slot_mappings)
        
        # Sample tokens
        # We need logits for the last valid token of each sequence
        # logits: [batch, max_len, vocab]
        last_indices = torch.tensor([l - 1 for l in seq_lengths], device=self.device)
        # Gather: [batch, 1, vocab]
        vocab_size = logits.size(-1)
        next_token_logits = logits.gather(1, last_indices.view(-1, 1, 1).expand(-1, -1, vocab_size)).squeeze(1)
        
        next_tokens = self._sample(next_token_logits, scheduler_output.scheduled_requests)
        
        # Update Rust scheduler
        token_updates = {
            req_id: next_tokens[i].item()
            for i, req_id in enumerate(scheduler_output.scheduled_requests)
        }
        self.rust_engine.update(token_updates)
        
        # Decode tokens
        results = {}
        for req_id, token_id in token_updates.items():
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self.active_requests[req_id]["output_tokens"].append(token_id)
            results[req_id] = token_text
        
        return results
    
    def _sample(
        self,
        logits: torch.Tensor,
        request_ids: List[str],
    ) -> torch.Tensor:
        """
        Sample next tokens using temperature, top-p, top-k.
        
        Args:
            logits: [batch_size, vocab_size]
            request_ids: List of request IDs for getting configs
            
        Returns:
            Sampled token IDs [batch_size]
        """
        # Apply temperature (using first request's config for simplicity)
        config = self.active_requests[request_ids[0]]["config"]
        logits = logits / config.temperature
        
        # Apply top-k
        if config.top_k > 0:
            top_k_values, _ = torch.topk(logits, config.top_k)
            logits[logits < top_k_values[:, -1:]] = float("-inf")
        
        # Apply top-p (nucleus sampling)
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        High-level generation API (blocking).
        
        Args:
            prompt: Input text
            generation_config: Generation parameters
            
        Returns:
            Generated text
        """
        request_id = self.add_request(prompt, generation_config)
        
        # Run steps until we have generated max_tokens
        max_tokens = generation_config.max_tokens if generation_config else 100
        
        # Keep stepping until we've generated enough tokens
        # Note: We use a safety multiplier to allow for steps that don't produce tokens
        max_steps = max_tokens * 10  # Safety limit to avoid infinite loops
        
        for step_idx in range(max_steps):
            results = self.step()
            
            # Check how many tokens we've generated so far
            num_generated = len(self.active_requests[request_id]["output_tokens"])
            
            # Stop if we've reached max_tokens
            if num_generated >= max_tokens:
                break
        
        # Decode full output
        output_tokens = self.active_requests[request_id]["output_tokens"]
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_text
