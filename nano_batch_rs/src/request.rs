#[derive(Clone)]
pub enum RequestStatus {
    Waiting,
    Preempted,
    Running,
    Finished,
}

#[derive(Clone)]
pub struct SamplingParams {
    #[allow(dead_code)]
    pub temperature: f32,
    #[allow(dead_code)]
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct Request {
    pub request_id: String,         // unique identifier
    pub prompt_token_ids: Vec<u32>, // initial input
    pub output_token_ids: Vec<u32>, // generated tokens
    pub status: RequestStatus,      // state
    pub sampling_params: SamplingParams,

    // blocks
    pub logical_blocks: Vec<u32>, // logical_block[i] = physical_block
    pub block_size: usize,
}

impl Request {
    /// Total number of tokens
    pub fn num_tokens(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Indicates whether any of the stopping conditions of the request are met.
    pub fn should_end(&self) -> bool {
        // check if max_tokens is reached
        if self.output_token_ids.len() >= self.sampling_params.max_tokens {
            return true;
        }

        // check if end_tokens are in the last position
        if let Some(&last_token) = self.output_token_ids.last() {
            if self.sampling_params.stop_tokens.contains(&last_token) {
                return true;
            }
        }

        false
    }

    /// Preempts if there is no more space in the currently allocated blocks.
    pub fn should_preempt(&self) -> bool {
        let current_capacity = self.logical_blocks.len() * self.block_size;

        self.num_tokens() + 1 > current_capacity
    }

    pub fn needed_blocks(&self) -> usize {
        match self.status {
            RequestStatus::Preempted => 1,
            // (a + b - 1) / b = ceil_div(a, b)
            RequestStatus::Waiting => (self.num_tokens() + self.block_size - 1) / self.block_size,
            _ => panic!("Should not check how many blocks finished or running requests need."),
        }
    }

    pub fn assign_physical_blocks(&mut self, physical_block_ids: Vec<u32>) {
        self.logical_blocks.extend(physical_block_ids);
        match self.status {
            RequestStatus::Waiting => self.status = RequestStatus::Running,
            RequestStatus::Preempted => self.status = RequestStatus::Running,
            _ => panic!("Should not be assigning blocks to finished or running requests."),
        }
    }
}
