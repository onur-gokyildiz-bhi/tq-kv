//! Paged KV Cache — PagedAttention-style block management.
//!
//! Divides KV cache into fixed-size blocks of N tokens.
//! A block table maps logical block indices to physical GPU memory.
//! Reduces memory waste from 60-80% to <4% (vLLM).
//!
//! With TurboQuant compression, blocks store compressed keys:
//!   block = [packed_indices: bytes_per_key × block_size][norms: f32 × block_size]

/// Default block size in tokens.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Physical block in GPU memory.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalBlock {
    /// Unique block ID.
    pub id: usize,
    /// Byte offset in the GPU memory pool.
    pub offset: usize,
    /// Whether this block is currently allocated.
    pub allocated: bool,
    /// Reference count (for copy-on-write with beam search).
    pub ref_count: u32,
}

/// Block table for a single sequence.
/// Maps logical block index → physical block ID.
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub sequence_id: usize,
    /// Logical block indices → physical block IDs.
    pub blocks: Vec<usize>,
    /// Number of tokens stored (may be less than blocks.len() * block_size).
    pub n_tokens: usize,
}

/// Paged KV cache block allocator.
///
/// Manages a pool of fixed-size physical blocks.
/// Sequences request blocks as they grow, release when done.
pub struct PagedKvAllocator {
    /// Block size in tokens.
    pub block_size: usize,
    /// Total number of physical blocks.
    pub n_blocks: usize,
    /// Physical block metadata.
    blocks: Vec<PhysicalBlock>,
    /// Free block IDs.
    free_list: Vec<usize>,
    /// Per-sequence block tables.
    pub sequences: std::collections::HashMap<usize, BlockTable>,
    /// Bytes per token in KV cache (depends on compression config).
    pub bytes_per_token: usize,
}

impl PagedKvAllocator {
    /// Create allocator with given GPU memory budget.
    pub fn new(
        total_gpu_bytes: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        bits: u8, // TQ compression bits (0 = fp16)
    ) -> Self {
        // Calculate bytes per token per layer
        let k_bytes = if bits > 0 {
            (head_dim * bits as usize + 7) / 8 + 4 // packed indices + f32 norm
        } else {
            head_dim * 2 // fp16
        };
        let v_bytes = head_dim * 2; // V always fp16 by default
        let bytes_per_token = n_kv_heads * (k_bytes + v_bytes);

        let block_bytes = block_size * bytes_per_token;
        let n_blocks = total_gpu_bytes / block_bytes;

        let blocks: Vec<PhysicalBlock> = (0..n_blocks)
            .map(|id| PhysicalBlock {
                id,
                offset: id * block_bytes,
                allocated: false,
                ref_count: 0,
            })
            .collect();
        let free_list: Vec<usize> = (0..n_blocks).rev().collect();

        Self {
            block_size,
            n_blocks,
            blocks,
            free_list,
            sequences: std::collections::HashMap::new(),
            bytes_per_token,
        }
    }

    /// Allocate a physical block. Returns block ID.
    pub fn alloc_block(&mut self) -> Option<usize> {
        let id = self.free_list.pop()?;
        self.blocks[id].allocated = true;
        self.blocks[id].ref_count = 1;
        Some(id)
    }

    /// Free a physical block (decrement ref count).
    pub fn free_block(&mut self, id: usize) {
        if id < self.n_blocks {
            self.blocks[id].ref_count = self.blocks[id].ref_count.saturating_sub(1);
            if self.blocks[id].ref_count == 0 {
                self.blocks[id].allocated = false;
                self.free_list.push(id);
            }
        }
    }

    /// Register a new sequence and allocate initial blocks.
    pub fn new_sequence(&mut self, seq_id: usize, initial_tokens: usize) -> Option<()> {
        let n_blocks_needed = (initial_tokens + self.block_size - 1) / self.block_size;
        let mut block_ids = Vec::with_capacity(n_blocks_needed);

        for _ in 0..n_blocks_needed {
            block_ids.push(self.alloc_block()?);
        }

        self.sequences.insert(seq_id, BlockTable {
            sequence_id: seq_id,
            blocks: block_ids,
            n_tokens: initial_tokens,
        });
        Some(())
    }

    /// Append tokens to a sequence, allocating new blocks as needed.
    pub fn append_tokens(&mut self, seq_id: usize, n_new_tokens: usize) -> Option<()> {
        let table = self.sequences.get(&seq_id)?;
        let old_total = table.n_tokens;
        let new_total = old_total + n_new_tokens;
        let old_blocks = (old_total + self.block_size - 1) / self.block_size;
        let new_blocks = (new_total + self.block_size - 1) / self.block_size;

        // Allocate new blocks first (avoids double-borrow on self)
        let mut new_block_ids = Vec::new();
        for _ in old_blocks..new_blocks {
            new_block_ids.push(self.alloc_block()?);
        }

        let table = self.sequences.get_mut(&seq_id)?;
        table.blocks.extend(new_block_ids);
        table.n_tokens = new_total;
        Some(())
    }

    /// Release all blocks for a sequence.
    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(table) = self.sequences.remove(&seq_id) {
            for block_id in table.blocks {
                self.free_block(block_id);
            }
        }
    }

    /// Number of free blocks remaining.
    pub fn n_free_blocks(&self) -> usize { self.free_list.len() }

    /// Maximum additional tokens that can be stored.
    pub fn free_token_capacity(&self) -> usize { self.free_list.len() * self.block_size }

    /// GPU utilization percentage.
    pub fn utilization(&self) -> f32 {
        let used = self.n_blocks - self.free_list.len();
        used as f32 / self.n_blocks as f32 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_alloc_basic() {
        // 1 MB pool, block_size=16, 8 kv heads, dim=128, 4-bit TQ
        let mut alloc = PagedKvAllocator::new(
            1_000_000, 16, 8, 128, 4,
        );
        assert!(alloc.n_free_blocks() > 0);

        alloc.new_sequence(0, 32).unwrap();
        let table = alloc.sequences.get(&0).unwrap();
        assert_eq!(table.n_tokens, 32);
        assert_eq!(table.blocks.len(), 2); // 32/16 = 2 blocks

        alloc.append_tokens(0, 16).unwrap();
        let table = alloc.sequences.get(&0).unwrap();
        assert_eq!(table.n_tokens, 48);
        assert_eq!(table.blocks.len(), 3); // 48/16 = 3 blocks

        let free_before = alloc.n_free_blocks();
        alloc.free_sequence(0);
        assert_eq!(alloc.n_free_blocks(), free_before + 3);
    }
}
