# nano_batch: High-Performance LLM Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

**nano_batch** is a minimalist, high-performance inference engine for Large Language Models, built from scratch. It combines a **Rust-based scheduler** backend with a **PagedAttention kernel** in PyTorch to achieve vastly improved throughput and latency relative to standard implementations such as HuggingFace Transformers `.generate()`.

## Key Features

- **Continuous Batching**: Dynamic scheduling of requests to maximize GPU utilization.
- **PagedAttention**: Memory-efficient KV cache management inspired by vLLM.
- **Rust Core**: High-performance scheduler and block allocator written in Rust.

## Performance

I benchmarked `nano_batch` against a standard HuggingFace baseline using a Mistral 7B model (dummy weights but same architecture). The results demonstrate significant improvements in both throughput and latency.

### 1. Throughput Comparison (4.4x Speedup)

The engine achieves **300+ tokens/s** compared to the baseline's ~70 tokens/s.

![Throughput Comparison](images/mistral_throughput.png)

### 2. Throughput Scaling

The real power of `nano_batch` lies in its ability to scale with concurrency. While the baseline plateaus, the engine's throughput increases with the number of concurrent requests, leveraging continuous batching.

![Throughput Scaling](images/mistral_scaling.png)

### 3. Latency Reduction

I achieve a **92% reduction in First Token Latency (TTFT)** and **77% lower Average Latency** per token, meaning a significantly snappier user experience even under load in theory.

![Latency Comparison](images/mistral_latency.png)

## TODO:
- More benchmarks
- Trition paged attention kernel (could properly use the slot mappings)
- Run on full weight mistral 7B model

## Acknowledgments

This project was greatly inspired and helped by:

- **[Mistral 7B Paper](https://arxiv.org/pdf/2310.06825)** - For the excellent model architecture and implementation details
- **[Paged Attention Paper](https://arxiv.org/pdf/2309.06180)** - For the innovative approach to memory-efficient attention
- **[vLLM Project](https://github.com/vllm-project/vllm)** - For pioneering continuous batching and production serving patterns (with a very well engineered codebase)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt
maturin develop --release

# Run benchmarks
python nano_batch_models/benchmarks/dummy_nano_batch_benchmark.py
```