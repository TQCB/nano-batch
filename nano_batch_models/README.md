# nano_batch_models

Mistral model implementations built on the `nano_batch` core scheduling engine.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         nano_batch_models                       │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Models            │  Inference        │   Utilities           │
│ ┌─────────────────┐ │ ┌───────────────┐ │ ┌───────────────────┐ │
│ │ mistral.py      │ │ │inference_eng. │ │ │ tokenizer.py      │ │
│ │ • MistralConfig │ │ │ • Inference   │ │ │ • MistralTokenizer│ │
│ │ • MistralForLM  │ │ │   Engine      │ │ │ • encode/decode   │ │
│ │ • MistralModel  │ │ │ • Generation  │ │ │ • batch ops       │ │
│ │ • MistralLayer  │ │ │   Config      │ │ └───────────────────┘ │
│ │ • MistralMLP    │ │ └───────────────┘ │                       │
│ └─────────────────┘ │                   │ ┌───────────────────┐ │
│                     │ ┌───────────────┐ │ │ utils.py          │ │
│ ┌─────────────────┐ │ │paged_mistral.py││ │ • load_mistral    │ │
│ │paged_mistral.py │ │ │ • PagedMistral│ │ │ • memory calc     │ │
│ │ • PagedMistral  │ │ │   Attention   │ │ │ • generate        │ │
│ │ • PagedAttention│ │ │ • PagedDecoder│ │ │ • masks           │ │
│ └─────────────────┘ │ └───────────────┘ │ └───────────────────┘ │
└─────────────────────┴───────────────────┴───────────────────────┘
                              │
                              │ Uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      nano_batch_rs (Rust Core)                  │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│ │ Engine          │ │ paged_attention │ │ KVCache             │ │
│ │ • add_request   │ │ • paged_attn    │ │ • write_kv          │ │
│ │ • step          │ │   fwd           │ │ • key_cache         │ │
│ │ • update        │ │ • paged_kv_cache│ │ • value_cache       │ │
│ │ • block_alloc   │ └─────────────────┘ └─────────────────────┘ │
│ └─────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Model Hierarchy

```
MistralForCausalLM                    PagedMistralForCausalLM
├── MistralModel                      ├── embed_tokens
│   ├── embed_tokens                  ├── layers (PagedMistralDecoderLayer)
│   ├── layers (MistralDecoderLayer)  │   ├── PagedMistralAttention
│   │   ├── MistralAttention          │   │   └── paged_attention_fwd
│   │   └── MistralMLP                │   └── MistralMLP
│   └── norm                          ├── norm
└── lm_head                           └── lm_head
       │                                     │
       │ converts weights                    │ uses
       ▼                                     ▼
    ┌────────────────────────────────────────────────┐
    │         InferenceEngine                        │
    │  ┌──────────────┐    ┌─────────────────┐       │
    │  │ Rust Engine  │◄──►│ PyTorch Model   │       │
    │  │ (Scheduler)  │    │ (PagedAttention)│       │
    │  └──────────────┘    └─────────────────┘       │
    │         │                    │                 │
    │         ▼                    ▼                 │
    │  ┌──────────────┐    ┌─────────────────┐       │
    │  │ Block Tables │    │ KV Caches       │       │
    │  │ Allocation   │    │ (per layer)     │       │
    │  └──────────────┘    └─────────────────┘       │
    └────────────────────────────────────────────────┘
```

## Quick Usage

```python
from nano_batch_models import (
    MistralForCausalLM,
    PagedMistralForCausalLM, 
    InferenceEngine,
    MistralTokenizer,
    GenerationConfig
)

# Load and convert model
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.2")
paged_model = PagedMistralForCausalLM.from_standard_mistral(model)

# Create inference engine
tokenizer = MistralTokenizer("mistralai/Mistral-7B-v0.2")
engine = InferenceEngine(paged_model, tokenizer, num_blocks=100, block_size=16)

# Generate text
output = engine.generate(
    "The future of AI is", 
    GenerationConfig(max_tokens=50, temperature=0.8)
)
print(output)
```

## Dependencies

- **nano_batch_rs**: Rust scheduling engine and PagedAttention kernels
- **torch**: PyTorch for model implementation
- **transformers**: HuggingFace model loading and tokenization