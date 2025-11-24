# nano_batch - Rust PagedAttention Scheduling Engine

A minimalist, high-performance **scheduling engine** for LLM inference with PagedAttention. This project focuses on the core systems programming challenge: efficient memory management and continuous batching in Rust.

## 🎯 Core Focus

This repository contains **only the essential scheduling primitives**:
- **Rust Block Allocator** - Memory-efficient block management with LRU eviction
- **Continuous Batching Scheduler** - Iteration-level scheduling for higher throughput  
- **PagedAttention Kernel** - Pure PyTorch implementation for KV cache access
- **KV Cache Management** - Physical block management

**For full model implementations** (Mistral 7B, inference engine, benchmarks), see [`nano_batch_models`](../nano_batch_models/).

## 🚀 Why This Matters

Modern LLM serving (vLLM, TensorRT-LLM) relies on sophisticated scheduling and memory management. This project demonstrates:
- ✅ **Rust for performance-critical code** - Type-safe scheduler with zero-cost abstractions
- ✅ **PagedAttention architecture** - Block-based KV cache eliminates fragmentation
- ✅ **Continuous batching** - Request-level scheduling for optimal GPU utilization
- ✅ **Clean FFI design** - Minimal, well-defined Python/Rust boundary

## 📦 Installation

### Prerequisites
- Rust (cargo) and Python 3.8+
- PyTorch 2.0+

### Build from Source

```bash
cd nano_batch_rs

# Install Python dependencies
pip install -r requirements.txt

# Build Rust extension
pip install maturin
maturin develop --release
```

## 💡 Quick Start

### Core Engine Demo

```python
from nano_batch import Engine, paged_attention_fwd, KVCache
import torch

# Initialize scheduler
engine = Engine(num_blocks=100, block_size=16)

# Add a request
engine.add_request(
    request_id="demo_1",
    prompt_token_ids=[1, 2, 3, 4, 5],
    max_tokens=50
)

# Run scheduling step
output = engine.step()
print(f"Scheduled: {output.scheduled_requests}")
print(f"Block tables: {output.block_tables}")
print(f"Slot mappings: {output.slot_mappings}")

# Initialize KV cache
kv_cache = KVCache(
    num_blocks=100,
    block_size=16,
    num_kv_heads=8,
    head_dim=128,
    device="cuda"
)

# Use PagedAttention kernel
# (see examples/core_engine_demo.py for full example)
```

### Run the Demo

```bash
python python/examples/core_engine_demo.py
```

## 📁 Project Structure

```
nano_batch_rs/
├── src/                          # Rust core (MAIN FOCUS)
│   ├── scheduler.rs             # Continuous batching scheduler
│   ├── block_allocator.rs       # Memory block management
│   ├── request.rs               # Request types
│   └── lib.rs                   # Python bindings (PyO3)
├── python/nano_batch/
│   ├── __init__.py              # Package exports
│   ├── engine.py                # Python wrapper for Rust
│   └── paged_attention.py       # PagedAttention kernel + KV cache
└── python/examples/
    └── core_engine_demo.py      # Minimal synthetic example
```

**What's NOT here:**
- ❌ Model implementations (Mistral, etc.) → See [`nano_batch_models/`](../nano_batch_models/)
- ❌ High-level inference APIs → See [`nano_batch_models/`](../nano_batch_models/)
- ❌ Tokenizers and model configs → See [`nano_batch_models/`](../nano_batch_models/)
- ❌ Benchmarks → See [`nano_batch_models/`](../nano_batch_models/)

## 🔧 API Reference

### Rust Engine (via Python)

```python
Engine(num_blocks: int, block_size: int)
```

- `add_request(request_id, prompt_token_ids, **params)` - Add inference request
- `step()` - Run one scheduling iteration
- `update(token_updates)` - Update with generated tokens

**Returns:** `SchedulerOutput` with:
- `scheduled_requests` - List of request IDs to process
- `block_tables` - Physical block mappings per request
- `slot_mappings` - Token → physical slot indices
- `num_tokens_per_request` - Token counts per request

### PagedAttention Kernel

```python
paged_attention_fwd(
    query,          # [batch, num_heads, 1, head_dim]
    key_cache,      # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache,    # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table,    # [batch, max_blocks]
    context_lens,   # [batch]
    block_size      # int
) -> torch.Tensor   # [batch, num_heads, 1, head_dim]
```

Pure PyTorch implementation (no Triton) - works on Windows, macOS, Linux.

### KV Cache

```python
KVCache(num_blocks, block_size, num_kv_heads, head_dim, dtype, device)
```

- `write_kv(keys, values, slot_mappings)` - Write to physical slots
- `get_kv_cache()` - Get full cache tensors
- `get_memory_usage()` - Memory statistics

## 🏗️ Architecture

```
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Rust Scheduler  │  ← Block allocation + request scheduling
│   (src/*.rs)    │
└────────┬────────┘
         │ SchedulerOutput(block_tables, slot_mappings)
         v
┌─────────────────┐
│  Python Model   │  ← Use with your own model
│  (user code)    │     OR nano_batch_models
└────────┬────────┘
         │ logits
         v
┌─────────────────┐
│ Token Sampling  │
└────────┬────────┘
         │ new_tokens
         v
┌─────────────────┐
│  Update Cache   │  ← write_kv(keys, values, slot_mappings)
└─────────────────┘
```

## 🧪 Testing

```bash
# Core engine test
python python/examples/core_engine_demo.py

# For full model tests, see nano_batch_models/
```

## 📊 Performance Characteristics

PagedAttention memory savings vs. naive KV cache:
- **60-80% reduction** in memory fragmentation
- **Higher throughput** via continuous batching
- **Lower latency** in decode phase

## 🔗 Related

- **[nano_batch_models](../nano_batch_models/)** - Full Mistral 7B implementation using this engine
- **[vLLM](https://github.com/vllm-project/vllm)** - Production-ready LLM serving (inspiration)
- **[PagedAttention Paper](https://arxiv.org/abs/2309.06180)** - Original research

## 📝 License

MIT License

## 🙏 Acknowledgments

Inspired by vLLM and the PagedAttention paper. This is a learning project demonstrating core architectural principles.

---

**Next Steps:**
1. Run the core demo: `python python/examples/core_engine_demo.py`
2. For full Mistral inference: See [`nano_batch_models/`](../nano_batch_models/)
3. Explore the Rust scheduler: `src/scheduler.rs`