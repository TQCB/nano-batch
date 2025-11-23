mod block_allocator;
mod request;
mod scheduler;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use block_allocator::BlockAllocator;
use request::{Request, RequestStatus, SamplingParams};
use scheduler::Scheduler;

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
    fn step(&mut self) -> scheduler::SchedulerOutput {
        self.scheduler.schedule()
    }

    fn convert_output_to_python(
        &self,
        output: scheduler::SchedulerOutput,
        py: Python,
    ) -> pyo3::Py<PyAny> {
        let result = PyDict::new(py);

        // scheduled_requests
        let scheduled_requests: Vec<String> = output.scheduled_requests;
        result
            .set_item("scheduled_requests", scheduled_requests)
            .unwrap();

        // block_tables - convert HashMap<String, Vec<u32>> to Python dict
        let block_tables_dict = PyDict::new(py);
        for (request_id, blocks) in output.block_tables {
            block_tables_dict.set_item(request_id, blocks).unwrap();
        }
        result.set_item("block_tables", block_tables_dict).unwrap();

        // slot_mappings
        let slot_mappings: Vec<u32> = output.slot_mappings;
        result.set_item("slot_mappings", slot_mappings).unwrap();

        // num_tokens_per_request - convert HashMap<String, usize> to Python dict
        let num_tokens_dict = PyDict::new(py);
        for (request_id, num_tokens) in output.num_tokens_per_request {
            num_tokens_dict.set_item(request_id, num_tokens).unwrap();
        }
        result
            .set_item("num_tokens_per_request", num_tokens_dict)
            .unwrap();

        result.into()
    }

    /// Use the tokens from the python worker to update our requests.
    fn update(&mut self, token_updates: HashMap<String, u32>) {
        // This is a simplified version - in reality you'd need more complex
        // logic to handle multiple tokens per request, etc.
        for (request_id, new_token) in token_updates {
            self.scheduler.update_request_token(&request_id, new_token);
        }
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
    fn step(&mut self, py: Python) -> pyo3::Py<PyAny> {
        let output = self.inner.step();
        self.inner.convert_output_to_python(output, py)
    }

    /// Update requests with newly generated tokens.
    fn update(&mut self, token_updates: HashMap<String, u32>) {
        self.inner.update(token_updates);
    }
}
