mod block_allocator;
mod request;
mod scheduler;

use std::collections::HashMap;

use pyo3::prelude::*;

use block_allocator::BlockAllocator;
use request::{Request, RequestStatus, SamplingParams};
use scheduler::Scheduler;

/// Type-safe Python interface for scheduler output
#[pyclass]
#[derive(Clone)]
pub struct PySchedulerOutput {
    #[pyo3(get)]
    pub scheduled_requests: Vec<String>,
    #[pyo3(get)]
    pub block_tables: HashMap<String, Vec<u32>>,
    #[pyo3(get)]
    pub slot_mappings: Vec<u32>,
    #[pyo3(get)]
    pub num_tokens_per_request: HashMap<String, usize>,
}

struct InnerEngine {
    scheduler: Scheduler,
}

impl InnerEngine {
    fn new(num_blocks: u32, block_size: u32) -> Self {
        let block_allocator = BlockAllocator::new(num_blocks, block_size);
        let scheduler = Scheduler::new(block_allocator);

        Self { scheduler }
    }

    /// Take a prompt and parameters from Python, creates a Request, and hands
    /// it to the scheduler.
    fn add_request(
        &mut self,
        request_id: String,
        prompt_token_ids: Vec<u32>,
        temperature: f32,
        top_p: f32,
        max_tokens: usize,
        stop_tokens: Vec<u32>,
        // block_size: usize
    ) {
        let sampling_params = SamplingParams {
            temperature,
            top_p,
            max_tokens,
            stop_tokens,
        };

        let request = Request {
            request_id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            status: RequestStatus::Waiting,
            sampling_params,
            logical_blocks: Vec::new(),
            block_size: self.scheduler.block_allocator.block_size as usize,
        };

        self.scheduler.add_request(request);
    }

    /// Runs a scheduling step and hands back a scheduler output.
    fn step(&mut self) -> PySchedulerOutput {
        let output = self.scheduler.schedule();
        PySchedulerOutput {
            scheduled_requests: output.scheduled_requests,
            block_tables: output.block_tables,
            slot_mappings: output.slot_mappings,
            num_tokens_per_request: output.num_tokens_per_request,
        }
    }

    /// Use the tokens from the python worker to update our requests.
    fn update(&mut self, token_updates: HashMap<String, u32>) {
        self.scheduler.update(token_updates);
    }
}

#[pyclass]
struct Engine {
    inner: InnerEngine,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(num_blocks: u32, block_size: u32) -> Self {
        Self {
            inner: InnerEngine::new(num_blocks, block_size),
        }
    }

    /// Add a new request to the scheduler.
    fn add_request(
        &mut self,
        request_id: String,
        prompt_token_ids: Vec<u32>,
        temperature: f32,
        top_p: f32,
        max_tokens: usize,
        stop_tokens: Vec<u32>,
    ) {
        self.inner.add_request(
            request_id,
            prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            stop_tokens,
        );
    }

    /// Run a scheduling step and return the batch information.
    fn step(&mut self) -> PySchedulerOutput {
        self.inner.step()
    }

    /// Update requests with newly generated tokens.
    fn update(&mut self, token_updates: HashMap<String, u32>) {
        self.inner.update(token_updates);
    }
}
