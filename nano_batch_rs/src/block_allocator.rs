use std::collections::HashSet;

pub struct BlockAllocator {
    #[allow(dead_code)]
    pub num_blocks: u32,
    #[allow(dead_code)]
    pub block_size: u32,
    pub free_blocks: Vec<u32>,
    pub allocated_blocks: HashSet<u32>,
}

impl BlockAllocator {
    pub fn new(num_blocks: u32, block_size: u32) -> BlockAllocator {
        let mut free_blocks: Vec<u32> = Vec::with_capacity(num_blocks as usize);

        for i in 0..num_blocks {
            free_blocks.push(i);
        }

        // reverse list of free blocks to store as stack we can pop off of
        let free_blocks = free_blocks.into_iter().rev().collect();

        BlockAllocator {
            num_blocks,
            block_size,
            free_blocks,
            allocated_blocks: HashSet::new(),
        }
    }

    /// Return Some(free_block_id) from off the top of the free_blocks stack
    /// Returns None if there are no free blocks left.
    ///
    /// Returned free block is inserted into set of allocated blocks.
    fn allocate(&mut self) -> Option<u32> {
        let free_block = self.free_blocks.pop()?;
        self.allocated_blocks.insert(free_block);

        Some(free_block)
    }

    pub fn allocate_multiple(&mut self, count: usize) -> Option<Vec<u32>> {
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            allocated.push(self.allocate()?)
        }

        Some(allocated)
    }

    /// Adds a block (that must be currently allocated) to free blocks,
    /// removing it from the set of allocated blocks.
    fn free(&mut self, block_id: u32) -> Result<(), String> {
        if !self.allocated_blocks.contains(&block_id) {
            return Err("Block not allocated or already freed".to_string());
        }

        self.allocated_blocks.remove(&block_id);
        self.free_blocks.push(block_id);

        Ok(())
    }

    /// Calls free on all provided block ids
    pub fn free_multiple(&mut self, block_ids: Vec<u32>) -> Result<(), String> {
        for block_id in block_ids {
            self.free(block_id)?;
        }

        Ok(())
    }
}
