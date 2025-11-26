# nano_batch: High-Performance LLM Inference Engine

**nano_batch** is a minimalist, high-performance inference engine for Large Language Models, built from scratch to demonstrate advanced systems engineering concepts. It combines a **Rust-based scheduler** with a custom **PagedAttention kernel** in PyTorch to achieve state-of-the-art throughput and latency.

## Key Features

- **Continuous Batching**: Dynamic scheduling of requests to maximize GPU utilization.
- **PagedAttention**: Memory-efficient KV cache management inspired by vLLM.
- **Rust Core**: High-performance scheduler and block allocator written in Rust.
- **Python Bindings**: Seamless integration with PyTorch models via PyO3.

## Performance

We benchmarked `nano_batch` against a standard HuggingFace baseline using a Mistral 7B model (dummy weights). The results demonstrate significant improvements in both throughput and latency.

### 1. Throughput Comparison (4.4x Speedup)

Our engine achieves **300+ tokens/s** compared to the baseline's ~70 tokens/s.

![Throughput Comparison](images/mistral_throughput.png)

### 2. Throughput Scaling

The true power of `nano_batch` lies in its ability to scale with concurrency. While the baseline plateaus, our engine's throughput increases with the number of concurrent requests, leveraging continuous batching.

![Throughput Scaling](images/mistral_scaling.png)

### 3. Latency Reduction

We achieve a **92% reduction in First Token Latency (TTFT)** and **77% lower Average Latency** per token. This ensures a snappy user experience even under load.

![Latency Comparison](images/mistral_latency.png)

## Architecture

- **`nano_batch_rs`**: Rust crate containing the `Scheduler`, `BlockAllocator`, and `Request` management.
- **`nano_batch_models`**: Python package with the `InferenceEngine`, `PagedMistral` model, and benchmarks.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt
maturin develop --release

# Run benchmarks
python nano_batch_models/benchmarks/dummy_nano_batch_benchmark.py
```

## Author

Built as a high-performance systems project to demonstrate expertise in ML infrastructure, Rust, and CUDA/PyTorch optimization.
