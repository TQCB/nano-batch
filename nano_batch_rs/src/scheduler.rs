use std::collections::VecDeque;

use crate::{
    request::Request,
    block_allocator::BlockAllocator,
};

struct SchedulerOutput {
    requests: Vec<Request>,
}

struct Queue {
    requests: VecDeque<Request>
}

/// Scheduler needs look at waiting and running requests at every step,
/// and decide which ones get to run based on available memory (blocks).
pub struct Scheduler {
    waiting: Queue,
    running: Queue,
    block_alloctor: BlockAllocator
}

impl Scheduler {
    /// Handles which requests to run.
    /// 
    /// Running jobs: the scheduler will make sure that each running job still
    /// has space for 1 more token to generate. If not, the request needs to be
    /// preempted and their associated blocks freed.
    /// 
    /// Waiting jobs: if there are free blocks, move waiting requests into running.
    fn schedule(&self) -> SchedulerOutput {
        for request in self.running.requests {
            if request.should_end() {
                // get blocks from request
                // free those blocks
                // remove request from running queue
            } else if request.should_preempt() {
                // check how many tokens we need to generate, and how much space
                // is left in the request's blocks
                // if there is no more space, we send the request
                // to the wait queue with queue.push_front()
            }
        };

        // Once we're done handling the running requests,
        // we check how many waiting requests we have.
        // For now we implement FCFS allocation where the FI request gets
        // allocated blocks to and added to running queue.
        
        let n_free_blocks = self.block_alloctor.n_free_blocks();
        for request in self.waiting.requests {
            let prefill_blocks_needed = request.prefill_len() / self.block_alloctor.block_size;
            if prefill_blocks_needed <= n_free_blocks {
                // if we have enough space to prefill,
                // then we prefill and add to running.
                n_free_blocks -= prefill_blocks_needed;
            }
        }

        SchedulerOutput { requests: () }
    }
}