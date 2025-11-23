use std::collections::{HashMap, VecDeque};

use crate::{
    block_allocator::BlockAllocator,
    request::{Request, RequestStatus},
};

pub struct SchedulerOutput {
    pub scheduled_requests: Vec<String>,         // request ids
    pub block_tables: HashMap<String, Vec<u32>>, // request id: physical block
    pub slot_mappings: Vec<u32>,                 // token index: physical slot
    pub num_tokens_per_request: HashMap<String, usize>,
}

struct Queue {
    requests: VecDeque<Request>,
}

/// Scheduler needs look at waiting and running requests at every step,
/// and decide which ones get to run based on available memory (blocks).
///
/// TODO: Could be worthwhile to keep track of the minimum amount of blocks
/// needed by a request. This would allow for early stopping when allocating
/// blocks during scheduling, if the minimum amount of blocks needed by the
/// waiting requests is greater than the amount of free blocks
pub struct Scheduler {
    waiting: Queue,
    running: Queue,
    pub block_allocator: BlockAllocator,
}

impl Scheduler {
    pub fn new(block_allocator: BlockAllocator) -> Self {
        Scheduler {
            waiting: Queue {
                requests: VecDeque::new(),
            },
            running: Queue {
                requests: VecDeque::new(),
            },
            block_allocator,
        }
    }

    pub fn add_request(&mut self, request: Request) {
        self.waiting.requests.push_back(request);
    }

    pub fn update_request_token(&mut self, request_id: &str, new_token: u32) {
        // Find the request in running requests and update it
        for request in &mut self.running.requests {
            if request.request_id == request_id {
                request.output_token_ids.push(new_token);
                return;
            }
        }
        // If not found in running, it might be in waiting queue
        for request in &mut self.waiting.requests {
            if request.request_id == request_id {
                request.output_token_ids.push(new_token);
                return;
            }
        }
    }

    /// Update multiple requests with newly generated tokens
    pub fn update(&mut self, token_updates: HashMap<String, u32>) {
        for (request_id, new_token) in token_updates {
            self.update_request_token(&request_id, new_token);
        }
    }

    fn generate_slot_mappings(
        scheduled_requests: &[Request],
        num_tokens_per_request: &HashMap<String, usize>,
    ) -> Vec<u32> {
        let mut slot_mappings = Vec::new();

        for request in scheduled_requests {
            let block_size = request.block_size;
            let num_tokens_to_process = *num_tokens_per_request
                .get(&request.request_id)
                .expect("Scheduled request not found in num_tokens_per_request map");

            // find starting token index
            // for prefill its 0
            // for decode its total tokens
            let start_token_idx_in_request = match request.status {
                RequestStatus::Preempted => 0,
                RequestStatus::Waiting => request.num_tokens(),
                _ => panic!("Should not be scheduling running or finished requests."),
            };

            for i in 0..num_tokens_to_process {
                let current_token_idx_in_request = start_token_idx_in_request + i;

                let logical_block_idx = current_token_idx_in_request / block_size;
                let offset_in_block = current_token_idx_in_request % block_size;

                if logical_block_idx >= request.logical_blocks.len() {
                    panic!(
                        "Logical block index {} out of bounds for request {}",
                        logical_block_idx, request.request_id
                    )
                }

                let physical_block_id = request.logical_blocks[logical_block_idx];
                let physical_slot_idx = (physical_block_id as usize * block_size) + offset_in_block;
                slot_mappings.push(physical_slot_idx as u32)
            }
        }

        slot_mappings
    }

    /// Handles which requests to run.
    ///
    /// Running jobs: First the scheduler checks if a request should end (end
    /// token, going past max_token amount) -- ending a request will free its
    /// blocks and set the request to finished. The scheduler will then make sure
    /// that each running job still has space for 1 more token to generate. If
    /// not, the request needs to be preempted: it will be added back to the
    /// waiting requests to be allocated a new block.
    ///
    /// Waiting jobs: if there are free blocks, then waiting requests need to be
    /// allocated. Running blocks that had to preempt (and thus become waiting)
    /// are given priority so they can continue running. If not, waiting requests
    /// prefill their inputs and become running.
    ///
    /// Once the above is handled, we return the current running and finished
    /// requests.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut scheduled_requests = Vec::new();
        let mut block_tables: HashMap<String, Vec<u32>> = HashMap::new();
        let mut num_tokens_per_request: HashMap<String, usize> = HashMap::new();
        let mut current_scheduled_requests: Vec<Request> = Vec::new();

        let mut i = 0;
        while i < self.running.requests.len() {
            let req = &mut self.running.requests[i];

            if req.should_end() {
                req.status = RequestStatus::Finished;
            } else if req.should_preempt() {
                req.status = RequestStatus::Preempted;
            }

            match req.status {
                RequestStatus::Finished => {
                    let finished_req = self
                        .running
                        .requests
                        .remove(i)
                        .expect("Should not fail to remove request");
                    self.block_allocator
                        .free_multiple(finished_req.logical_blocks)
                        .expect("Blocks to freed should neither already be freed, or unallocated");
                    continue; // removed item so don't increment
                }
                RequestStatus::Preempted => {
                    let preempted_req = self
                        .running
                        .requests
                        .remove(i)
                        .expect("Should not fail to remove request");
                    self.waiting.requests.push_front(preempted_req); // we add the preempted req to the front of the waiting list, so that it gets served next
                    continue; // removed item so don't increment
                }
                _ => i += 1, // move to next if neither finished or preempted
            }
        }

        // Once we're done handling the running requests,
        // we check how many waiting requests we have.
        // For now we implement FCFS allocation where the FI request gets
        // allocated blocks to and added to running queue.

        let n_free_blocks = self.block_allocator.free_blocks.len();

        // we risk deadlocking here if all running requests are preempted and no
        // blocks are left. First we check if there are no more running requests,
        // and then we check we have no more more free blocks. If this is true,
        // then we evict the oldest requests until n_free_blocks > 0
        if self.running.requests.len() == 0 && n_free_blocks == 0 {
            let mut victims = Vec::new();
            while n_free_blocks == 0 {
                let mut victim = self.waiting.requests.pop_back()
                    .expect("Should have been able to pop request of waiting list if there are 1. None running and 2. No more free blocks.");
                self.block_allocator
                    .free_multiple(victim.logical_blocks.clone())
                    .expect("Blocks to freed should neither already be freed, or unallocated");

                victim.logical_blocks.clear();

                victims.push(victim);
            }
            // once we have free blocks, we add the victims back to waiting
            // in reverse order so FIFO is conserved
            self.waiting.requests.extend(victims.into_iter().rev());
        }

        while let Some(mut request) = self.waiting.requests.pop_front() {
            let blocks_needed = request.needed_blocks();

            if n_free_blocks >= blocks_needed {
                let allocated_blocks = self.allocate_blocks_for_request(&request);
                request.assign_physical_blocks(allocated_blocks.clone());
                request.status = RequestStatus::Running;

                scheduled_requests.push(request.request_id.clone());
                block_tables.insert(request.request_id.clone(), allocated_blocks);
                num_tokens_per_request.insert(request.request_id.clone(), request.num_tokens());
                current_scheduled_requests.push(request.clone()); // clone for slot

                self.running.requests.push_back(request);
            } else {
                // not enough blocks, put it back and stop
                self.waiting.requests.push_front(request);
                break;
            }
        }

        // generate slot mappings
        let slot_mappings =
            Scheduler::generate_slot_mappings(&current_scheduled_requests, &num_tokens_per_request);

        SchedulerOutput {
            scheduled_requests,
            block_tables,
            slot_mappings,
            num_tokens_per_request,
        }
    }

    fn allocate_blocks_for_request(&mut self, request: &Request) -> Vec<u32> {
        let blocks_needed = request.needed_blocks();
        let physical_blocks = match self.block_allocator.allocate_multiple(blocks_needed) {
            Some(blocks) => blocks,
            None => return Vec::new(),
        };

        if physical_blocks.len() < blocks_needed {
            return Vec::new();
        }

        physical_blocks
    }
}
