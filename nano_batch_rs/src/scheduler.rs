use std::collections::VecDeque;

use crate::{
    request::{Request, Status},
    block_allocator::BlockAllocator,
};

struct SchedulerOutput {
    running: Vec<Request>,
    finished: Vec<Request>
}

struct Queue {
    requests: VecDeque<Request>
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
    block_alloctor: BlockAllocator
}

impl Scheduler {
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
    fn schedule(&self) -> SchedulerOutput {
        for request in self.running.requests {
            if request.should_end() {
                // get blocks from request
                // free those blocks
                // remove request from running queue
            } else if request.should_preempt() {
                // check how many tokens we need to generate, and how much space
                // is left in the request's blocks.
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
            match request.status {
                Status::WaitingDecode => {

                },
                Status::WaitingPrefill => {
                    
                },
                _ => panic!("Running or finished job found in waiting queue.")
            }
        }

        SchedulerOutput { running: (), finished: ()}
    }
}
