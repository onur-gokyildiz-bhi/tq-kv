//! Tree data structure + CPU reference attention for EAGLE speculative decode.
//!
//! # Why the CPU reference comes first
//!
//! Sprint 2's deliverable is a GPU tree-attention kernel. Before wiring a new
//! kernel path into the target's decode loop, we lock in correctness on the
//! CPU — the mask builder, the per-query softmax scope, and the GQA head
//! folding. Once `tree_attention_cpu` matches hand-computed expectations on
//! the unit tests below, any GPU output that disagrees is a kernel bug.
//!
//! # The tree
//!
//! Nodes are draft-proposed tokens, indexed `0..N_nodes` in topological order
//! (a node's parent has a smaller index). Each node points to its parent (or
//! `None` for the root). Depth-D tree with branching factor B has
//! `sum(B^d for d=1..=D)` nodes — typical EAGLE settings use D=4-6 with
//! B=2-4, yielding 31 to 256+ candidates. This module caps at 64 so the
//! ancestor bitmask fits in one `u64` per row (64 candidates × 64 bits =
//! 512 B total, rounded up to kernel arg size).
//!
//! # Mask semantics
//!
//! For each node `i`, `ancestor_mask[i]` has bit `j` set iff node `j` is
//! `i` or one of its ancestors. Tree attention for query `i` then attends to:
//!
//! 1. Every committed prefix token (`0..seq_prefix`)  — unconditional
//! 2. Every new token `j` where `ancestor_mask[i]` has bit `j` set
//!
//! This preserves causality within the tree while letting siblings ignore
//! each other — the whole point of the tree structure.

use std::fmt;

/// Upper bound on candidates per tree. Driven by the `u64` ancestor mask.
pub const MAX_TREE_NODES: usize = 64;

/// Static description of a tree of draft-proposed tokens.
#[derive(Debug, Clone)]
pub struct TreeSpec {
    /// Parent index per node. `None` marks a root (typically just index 0).
    /// Must satisfy `parents[i] < Some(i)` for all non-root `i`
    /// (topological order).
    pub parents: Vec<Option<usize>>,
    /// Depth per node. 0 for roots, `depth[parent]+1` otherwise.
    pub depths: Vec<u32>,
}

impl TreeSpec {
    pub fn n_nodes(&self) -> usize { self.parents.len() }

    /// Linear chain `0 → 1 → 2 → ... → n-1`. Every node attends to all
    /// ancestors, so tree attention on a chain degenerates to plain causal
    /// attention — used as the canonical equivalence check against
    /// `flash_decode`.
    pub fn linear(n: usize) -> Self {
        let mut parents = Vec::with_capacity(n);
        let mut depths  = Vec::with_capacity(n);
        for i in 0..n {
            if i == 0 {
                parents.push(None);
                depths.push(0);
            } else {
                parents.push(Some(i - 1));
                depths.push(i as u32);
            }
        }
        Self { parents, depths }
    }

    /// Full balanced tree of given branching factor and depth. Layer-major
    /// node indexing: root = 0, layer-1 = 1..=B, layer-2 = B+1..=B+B^2, etc.
    /// Rejects configurations that would exceed `MAX_TREE_NODES`.
    pub fn full(branching: u32, depth: u32) -> Result<Self, TreeError> {
        if branching == 0 {
            return Err(TreeError::Msg("branching must be >= 1".into()));
        }
        // Count nodes: 1 + B + B^2 + ... + B^depth = (B^(depth+1) - 1) / (B - 1)
        // for B > 1; (depth+1) for B=1.
        let total: usize = (0..=depth).map(|d| branching.pow(d) as usize).sum();
        if total > MAX_TREE_NODES {
            return Err(TreeError::Msg(format!(
                "full(B={},D={}) → {} nodes exceeds MAX_TREE_NODES={}",
                branching, depth, total, MAX_TREE_NODES
            )));
        }

        let mut parents = vec![None; total];
        let mut depths = vec![0u32; total];
        // Assign children layer-by-layer. Layer d has B^d nodes. Layer d+1 has
        // B^(d+1). Layer-d node index range starts at cumulative(d).
        let mut cum = vec![0usize; (depth + 2) as usize];
        for d in 0..=depth {
            cum[(d + 1) as usize] = cum[d as usize] + branching.pow(d) as usize;
        }
        for d in 1..=depth {
            let layer_start = cum[d as usize];
            let layer_end   = cum[(d + 1) as usize];
            let parent_layer_start = cum[(d - 1) as usize];
            for (k, i) in (layer_start..layer_end).enumerate() {
                let parent_idx = parent_layer_start + k / branching as usize;
                parents[i] = Some(parent_idx);
                depths[i] = d;
            }
        }
        Ok(Self { parents, depths })
    }

    /// Validate the invariants the mask builder + GPU kernel will rely on.
    pub fn validate(&self) -> Result<(), TreeError> {
        let n = self.n_nodes();
        if n == 0 {
            return Err(TreeError::Msg("empty tree".into()));
        }
        if n > MAX_TREE_NODES {
            return Err(TreeError::Msg(format!(
                "{} nodes exceeds MAX_TREE_NODES={}", n, MAX_TREE_NODES
            )));
        }
        if self.depths.len() != n {
            return Err(TreeError::Msg(format!(
                "depths len {} != n_nodes {}", self.depths.len(), n
            )));
        }
        for i in 0..n {
            match self.parents[i] {
                None => {
                    if self.depths[i] != 0 {
                        return Err(TreeError::Msg(format!(
                            "root node {} has depth {} (expected 0)",
                            i, self.depths[i]
                        )));
                    }
                }
                Some(p) => {
                    if p >= i {
                        return Err(TreeError::Msg(format!(
                            "node {} has parent {} — topological order requires parent < child",
                            i, p
                        )));
                    }
                    if self.depths[i] != self.depths[p] + 1 {
                        return Err(TreeError::Msg(format!(
                            "node {} depth {} != parent {} depth {} + 1",
                            i, self.depths[i], p, self.depths[p]
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum TreeError {
    Msg(String),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { TreeError::Msg(s) => write!(f, "{}", s) }
    }
}

impl std::error::Error for TreeError {}

/// Build a per-node ancestor bitmask: `bit j` of `mask[i]` is set iff node
/// `j` is an ancestor of `i` OR `j == i` itself.
///
/// Assumes `spec.validate()` has already succeeded. Correct only because
/// parents are in topological order: every `parent[i]` has its mask fully
/// populated by the time we reach `i`.
pub fn build_ancestor_mask(spec: &TreeSpec) -> Result<Vec<u64>, TreeError> {
    spec.validate()?;
    let n = spec.n_nodes();
    let mut mask = vec![0u64; n];
    for i in 0..n {
        let self_bit = 1u64 << i;
        let inherited = match spec.parents[i] {
            None    => 0,
            Some(p) => mask[p],
        };
        mask[i] = self_bit | inherited;
    }
    Ok(mask)
}

/// CPU reference: tree-masked attention output for a batch of queries.
///
/// Shape layout (all row-major, contiguous):
/// - `q`           : `[n_nodes, n_heads, head_dim]`
/// - `k`, `v`      : `[n_kv_heads, seq_prefix + n_nodes, head_dim]`
///                   The first `seq_prefix` positions are the committed
///                   target prefix; the remaining `n_nodes` slots hold the
///                   draft-proposed K/V (one slot per tree node, in the
///                   same order as the tree's node indices).
/// - `ancestor_mask`: `[n_nodes]`, from [`build_ancestor_mask`].
///
/// Returns `[n_nodes, n_heads, head_dim]`. For each `(i, h)` the output is:
///
/// ```text
///   scores[p in 0..seq_prefix]           = (q[i,h] · k[kv_h,p])    * scale
///   scores[seq_prefix + j] for j in mask = (q[i,h] · k[kv_h,p'])   * scale
///   softmax over the selected positions
///   out[i,h] = sum_p weight[p] * v[kv_h, p]
/// ```
///
/// Where `kv_h = h / (n_heads / n_kv_heads)` is the standard GQA head fold.
pub fn tree_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ancestor_mask: &[u64],
    n_nodes: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_prefix: usize,
    scale: f32,
) -> Result<Vec<f32>, TreeError> {
    if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(TreeError::Msg(format!(
            "head dims invalid: n_heads={} n_kv_heads={}", n_heads, n_kv_heads
        )));
    }
    let group_size = n_heads / n_kv_heads;

    let q_expected = n_nodes * n_heads * head_dim;
    let kv_expected = n_kv_heads * (seq_prefix + n_nodes) * head_dim;
    if q.len() != q_expected {
        return Err(TreeError::Msg(format!("q len {} != {}", q.len(), q_expected)));
    }
    if k.len() != kv_expected || v.len() != kv_expected {
        return Err(TreeError::Msg(format!(
            "k/v len {}/{} != {}", k.len(), v.len(), kv_expected
        )));
    }
    if ancestor_mask.len() != n_nodes {
        return Err(TreeError::Msg(format!(
            "ancestor_mask len {} != n_nodes {}", ancestor_mask.len(), n_nodes
        )));
    }

    let seq_total = seq_prefix + n_nodes;
    let mut out = vec![0f32; q_expected];

    for i in 0..n_nodes {
        let mask_i = ancestor_mask[i];
        for h in 0..n_heads {
            let kv_h = h / group_size;
            let q_off = (i * n_heads + h) * head_dim;
            let q_row = &q[q_off .. q_off + head_dim];

            // Pass 1: compute scores at all in-scope positions, track max.
            // scores storage: seq_prefix entries for prefix + up to n_nodes
            // entries for the masked tree positions. For simplicity we
            // allocate seq_total and fill only the in-scope ones; then a
            // per-position `in_scope` bool tracks what to include in softmax.
            let mut scores  = vec![0f32; seq_total];
            let mut in_scope = vec![false; seq_total];
            let mut max_s   = f32::NEG_INFINITY;

            // Prefix: always attended.
            for p in 0..seq_prefix {
                let k_off = (kv_h * seq_total + p) * head_dim;
                let s = dot(q_row, &k[k_off .. k_off + head_dim]) * scale;
                scores[p] = s;
                in_scope[p] = true;
                if s > max_s { max_s = s; }
            }
            // Tree: only positions whose bit is set in mask_i.
            for j in 0..n_nodes {
                if (mask_i >> j) & 1 == 0 { continue; }
                let p = seq_prefix + j;
                let k_off = (kv_h * seq_total + p) * head_dim;
                let s = dot(q_row, &k[k_off .. k_off + head_dim]) * scale;
                scores[p] = s;
                in_scope[p] = true;
                if s > max_s { max_s = s; }
            }

            // Pass 2: exp + normalise.
            let mut denom = 0f32;
            for p in 0..seq_total {
                if !in_scope[p] { continue; }
                let e = (scores[p] - max_s).exp();
                scores[p] = e;
                denom += e;
            }
            if denom == 0.0 {
                // Can happen only if no position was in scope — that means
                // seq_prefix == 0 AND mask_i == 0, which means node `i` isn't
                // even a descendant of itself, which would be a validate()
                // bug. Guard anyway to avoid NaN propagation.
                continue;
            }
            let inv = 1.0 / denom;

            // Pass 3: weighted V sum.
            let out_off = q_off;  // same shape as q
            for p in 0..seq_total {
                if !in_scope[p] { continue; }
                let w = scores[p] * inv;
                let v_off = (kv_h * seq_total + p) * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] += w * v[v_off + d];
                }
            }
        }
    }

    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

// ─── GPU integration (Sprint 2 Day 2) ──────────────────────────────────────
//
// The CPU reference above has unit-test coverage of the mask logic and the
// attention numerics. Day 2 ports the inner loop to CUDA (`tree_decode_partial_f32`
// in `kernels/flash_decode.cu`, launched via `crate::cuda::kernels::tree_decode_partial`).
// The GPU test below asserts element-wise parity with `tree_attention_cpu`
// to within 1e-4.

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use super::*;
    use crate::cuda::kernels as K;
    use crate::cuda::kernels::KernelRegistry;
    use cudarc::driver::CudaContext;

    /// Construct a fresh registry for this test. Don't go through the global
    /// registry so parallel tests don't race on initialisation order.
    fn build_registry() -> Option<std::sync::Arc<KernelRegistry>> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let (sm_major, sm_minor) = {
            use cudarc::driver::sys;
            let mut device: sys::CUdevice = 0;
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            unsafe {
                sys::cuDeviceGet(&mut device, 0);
                sys::cuDeviceGetAttribute(&mut major,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
                sys::cuDeviceGetAttribute(&mut minor,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            }
            (major as u32, minor as u32)
        };
        KernelRegistry::new(&ctx, &stream, sm_major, sm_minor)
            .ok().map(std::sync::Arc::new)
    }

    /// Canonical test: 7-node binary tree, Qwen2-shaped heads (GQA 4/1),
    /// short prefix. Asserts GPU output matches CPU reference on every
    /// element to within 1e-4 absolute.
    #[test]
    fn tree_gpu_matches_cpu_on_binary_tree() {
        let reg = match build_registry() {
            Some(r) => r,
            None => { eprintln!("[skip] no CUDA device"); return; }
        };
        let reg = reg.as_ref();

        let tree = TreeSpec::full(2, 2).unwrap();
        let n_nodes = tree.n_nodes();      // 7
        let mask = build_ancestor_mask(&tree).unwrap();

        let n_heads = 4;
        let n_kv = 1;       // 4/1 GQA fold
        let head_dim = 16;  // small, keeps partial_o buffer tiny
        let seq_prefix = 5;
        let seq_kv = seq_prefix + n_nodes;
        let max_seq = seq_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let split_size = 8;
        let n_splits = (seq_kv + split_size - 1) / split_size;

        // Deterministic pseudo-random tensors (value depends only on index).
        let rand = |seed: u32, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761u32) ^ seed;
                let u = (x & 0xFFFFFF) as f32 / 16_777_216.0;
                (u - 0.5) * 2.0 * scale
            }).collect()
        };

        let q_host = rand(0x1234, n_nodes * n_heads * head_dim, 1.0);
        let k_host = rand(0xABCD, n_kv * seq_kv * head_dim, 0.5);
        let v_host = rand(0x9876, n_kv * seq_kv * head_dim, 0.3);

        // ── CPU reference ──
        let out_cpu = tree_attention_cpu(
            &q_host, &k_host, &v_host, &mask,
            n_nodes, n_heads, n_kv, head_dim, seq_prefix, scale,
        ).unwrap();

        // ── GPU path ──
        let stream = reg.stream.clone();
        let q_dev = stream.memcpy_stod(&q_host).unwrap();
        let k_dev = stream.memcpy_stod(&k_host).unwrap();
        let v_dev = stream.memcpy_stod(&v_host).unwrap();
        let mask_dev = stream.memcpy_stod(&mask).unwrap();
        let mut partial_o   = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits * head_dim).unwrap();
        let mut partial_max = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits).unwrap();
        let mut partial_sum = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits).unwrap();
        let mut out_dev     = stream.alloc_zeros::<f32>(n_nodes * n_heads * head_dim).unwrap();

        K::tree_decode_partial(
            reg, &q_dev, &k_dev, &v_dev, &mask_dev,
            &mut partial_o, &mut partial_max, &mut partial_sum,
            n_nodes, n_heads, n_kv, seq_prefix, seq_kv, head_dim, scale,
            split_size, max_seq,
        ).unwrap();
        K::flash_decode_reduce(
            reg, &partial_o, &partial_max, &partial_sum,
            &mut out_dev, n_heads, n_splits, head_dim, n_nodes,
        ).unwrap();
        stream.synchronize().unwrap();
        let out_gpu = stream.memcpy_dtov(&out_dev).unwrap();

        // Element-wise compare.
        let mut max_diff = 0f32;
        let mut argmax = 0;
        for i in 0..out_cpu.len() {
            let diff = (out_cpu[i] - out_gpu[i]).abs();
            if diff > max_diff { max_diff = diff; argmax = i; }
        }
        assert!(max_diff < 1e-4,
            "tree GPU vs CPU diverged: max |diff| = {:.2e} at index {} (cpu={} gpu={})",
            max_diff, argmax, out_cpu[argmax], out_gpu[argmax]);
    }

    /// Degenerate case: linear chain behaves like plain causal decode,
    /// so each query's output must equal the naive all-in-scope attention
    /// (up to numerical rounding).
    #[test]
    fn tree_gpu_linear_chain_matches_naive() {
        let reg = match build_registry() {
            Some(r) => r,
            None => { eprintln!("[skip] no CUDA device"); return; }
        };
        let reg = reg.as_ref();

        let tree = TreeSpec::linear(4);
        let mask = build_ancestor_mask(&tree).unwrap();
        let n_nodes = 4;
        let n_heads = 2;
        let n_kv = 2;
        let head_dim = 8;
        let seq_prefix = 3;
        let seq_kv = seq_prefix + n_nodes;
        let max_seq = seq_kv;
        let scale = 0.5;
        let split_size = 4;

        let rand = |seed: u32, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761u32) ^ seed;
                let u = (x & 0xFFFFFF) as f32 / 16_777_216.0;
                (u - 0.5) * 2.0 * scale
            }).collect()
        };
        let q_host = rand(0x55, n_nodes * n_heads * head_dim, 1.0);
        let k_host = rand(0xAA, n_kv * seq_kv * head_dim, 0.4);
        let v_host = rand(0xCC, n_kv * seq_kv * head_dim, 0.3);

        let out_cpu = tree_attention_cpu(
            &q_host, &k_host, &v_host, &mask,
            n_nodes, n_heads, n_kv, head_dim, seq_prefix, scale,
        ).unwrap();

        let n_splits = (seq_kv + split_size - 1) / split_size;
        let stream = reg.stream.clone();
        let q_dev = stream.memcpy_stod(&q_host).unwrap();
        let k_dev = stream.memcpy_stod(&k_host).unwrap();
        let v_dev = stream.memcpy_stod(&v_host).unwrap();
        let mask_dev = stream.memcpy_stod(&mask).unwrap();
        let mut partial_o   = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits * head_dim).unwrap();
        let mut partial_max = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits).unwrap();
        let mut partial_sum = stream.alloc_zeros::<f32>(n_nodes * n_heads * n_splits).unwrap();
        let mut out_dev     = stream.alloc_zeros::<f32>(n_nodes * n_heads * head_dim).unwrap();

        K::tree_decode_partial(
            reg, &q_dev, &k_dev, &v_dev, &mask_dev,
            &mut partial_o, &mut partial_max, &mut partial_sum,
            n_nodes, n_heads, n_kv, seq_prefix, seq_kv, head_dim, scale,
            split_size, max_seq,
        ).unwrap();
        K::flash_decode_reduce(
            reg, &partial_o, &partial_max, &partial_sum,
            &mut out_dev, n_heads, n_splits, head_dim, n_nodes,
        ).unwrap();
        stream.synchronize().unwrap();
        let out_gpu = stream.memcpy_dtov(&out_dev).unwrap();

        for i in 0..out_cpu.len() {
            let diff = (out_cpu[i] - out_gpu[i]).abs();
            assert!(diff < 1e-4,
                "linear chain mismatch at {}: cpu={} gpu={} diff={:.2e}",
                i, out_cpu[i], out_gpu[i], diff);
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── TreeSpec validation ────────────────────────────────────────────────
    #[test]
    fn linear_tree_is_valid() {
        let t = TreeSpec::linear(5);
        t.validate().expect("linear(5) valid");
        assert_eq!(t.n_nodes(), 5);
        assert_eq!(t.parents, vec![None, Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(t.depths,  vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn full_binary_tree_layer_order() {
        // B=2, D=2 → 1 + 2 + 4 = 7 nodes, layer-major indexing
        let t = TreeSpec::full(2, 2).expect("full(2,2) valid");
        t.validate().expect("validate");
        assert_eq!(t.n_nodes(), 7);
        // Depths: [0, 1, 1, 2, 2, 2, 2]
        assert_eq!(t.depths, vec![0, 1, 1, 2, 2, 2, 2]);
        // Parents:
        //   root 0
        //   layer 1: nodes 1, 2 — both point to 0
        //   layer 2: nodes 3, 4 point to 1; nodes 5, 6 point to 2
        assert_eq!(t.parents[0], None);
        assert_eq!(t.parents[1], Some(0));
        assert_eq!(t.parents[2], Some(0));
        assert_eq!(t.parents[3], Some(1));
        assert_eq!(t.parents[4], Some(1));
        assert_eq!(t.parents[5], Some(2));
        assert_eq!(t.parents[6], Some(2));
    }

    #[test]
    fn full_tree_rejects_over_capacity() {
        // B=4, D=3 → 1 + 4 + 16 + 64 = 85 > 64
        let err = TreeSpec::full(4, 3).err().expect("should overflow");
        let s = err.to_string();
        assert!(s.contains("exceeds"), "got: {}", s);
    }

    #[test]
    fn bad_topological_order_rejected() {
        // Child points to a node that hasn't been seen yet.
        let t = TreeSpec {
            parents: vec![None, Some(2), Some(0)],
            depths:  vec![0, 2, 1],
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn depth_mismatch_rejected() {
        let t = TreeSpec {
            parents: vec![None, Some(0)],
            depths:  vec![0, 5],  // should be 1
        };
        assert!(t.validate().is_err());
    }

    // ── Mask semantics ─────────────────────────────────────────────────────
    #[test]
    fn linear_chain_mask_is_prefix_bits() {
        // In a chain, node i's ancestors are {0,1,...,i}, so the mask bit
        // pattern is (1<<(i+1)) - 1.
        let t = TreeSpec::linear(5);
        let m = build_ancestor_mask(&t).unwrap();
        for i in 0..5 {
            let expected: u64 = (1u64 << (i + 1)) - 1;
            assert_eq!(m[i], expected, "node {}: mask {:b} vs expected {:b}",
                       i, m[i], expected);
        }
    }

    #[test]
    fn binary_tree_mask_selects_path() {
        // Nodes 3, 4 are children of 1; node 1 is child of 0.
        // Node 3's ancestors in the tree: {0, 1, 3}. Mask bits: 0, 1, 3.
        // Node 4 similarly: {0, 1, 4}. Bits 0, 1, 4.
        // Node 6's ancestors: {0, 2, 6}. Bits 0, 2, 6.
        let t = TreeSpec::full(2, 2).unwrap();
        let m = build_ancestor_mask(&t).unwrap();
        assert_eq!(m[0], 0b0000001);
        assert_eq!(m[1], 0b0000011);
        assert_eq!(m[2], 0b0000101);
        assert_eq!(m[3], 0b0001011);
        assert_eq!(m[4], 0b0010011);
        assert_eq!(m[5], 0b0100101);
        assert_eq!(m[6], 0b1000101);
    }

    // ── Attention degenerates correctly ────────────────────────────────────
    #[test]
    fn single_node_tree_matches_naive_single_step() {
        // N=1, seq_prefix=3, the single node attends to prefix + itself.
        let head_dim = 4;
        let n_heads = 2;
        let n_kv = 2;
        let prefix = 3;
        let n_nodes = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Deterministic tensors
        let q: Vec<f32> = (0..n_nodes * n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k: Vec<f32> = (0..n_kv * (prefix + n_nodes) * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.05).collect();
        let v: Vec<f32> = (0..n_kv * (prefix + n_nodes) * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.03).collect();
        let mask = vec![0b1u64];

        let out = tree_attention_cpu(
            &q, &k, &v, &mask,
            n_nodes, n_heads, n_kv, head_dim, prefix, scale,
        ).unwrap();

        // Compare against a hand-rolled attention for the single query.
        let expected = naive_attention(&q, &k, &v,
                                       n_heads, n_kv, head_dim,
                                       prefix + n_nodes, scale);
        for i in 0..out.len() {
            assert!((out[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: {} vs {}", i, out[i], expected[i]);
        }
    }

    #[test]
    fn linear_chain_matches_causal_per_query() {
        // Each query i in a chain attends to prefix + {0..=i}. For each i
        // we can build a naive single-query attention over that sub-range
        // and compare.
        let head_dim = 4;
        let n_heads = 2;
        let n_kv = 1;  // GQA: 2 query heads share 1 KV head
        let prefix = 2;
        let n_nodes = 4;
        let scale = 0.5;

        let q: Vec<f32> = (0..n_nodes * n_heads * head_dim)
            .map(|i| ((i as f32 % 7.0) - 3.5) * 0.2).collect();
        let k_v_len = n_kv * (prefix + n_nodes) * head_dim;
        let k: Vec<f32> = (0..k_v_len).map(|i| ((i as f32 % 5.0) - 2.5) * 0.1).collect();
        let v: Vec<f32> = (0..k_v_len).map(|i| ((i as f32 % 11.0) - 5.5) * 0.07).collect();

        let tree = TreeSpec::linear(n_nodes);
        let mask = build_ancestor_mask(&tree).unwrap();
        let out = tree_attention_cpu(
            &q, &k, &v, &mask,
            n_nodes, n_heads, n_kv, head_dim, prefix, scale,
        ).unwrap();

        // Reference: for each node i, build a naive attention where K/V is
        // truncated to prefix + (i + 1) positions.
        for i in 0..n_nodes {
            let kv_len = prefix + i + 1;
            // Slice K/V to [n_kv, kv_len, head_dim] — need to re-layout because
            // rows are strided by (prefix + n_nodes), not kv_len.
            let k_trunc = truncate_kv(&k, n_kv, prefix + n_nodes, head_dim, kv_len);
            let v_trunc = truncate_kv(&v, n_kv, prefix + n_nodes, head_dim, kv_len);
            // Query slice for node i
            let q_off = i * n_heads * head_dim;
            let q_i = &q[q_off .. q_off + n_heads * head_dim];
            let expected_i = naive_attention(q_i, &k_trunc, &v_trunc,
                                             n_heads, n_kv, head_dim, kv_len, scale);
            for h in 0..n_heads {
                for d in 0..head_dim {
                    let out_idx = (i * n_heads + h) * head_dim + d;
                    let exp_idx = h * head_dim + d;
                    let diff = (out[out_idx] - expected_i[exp_idx]).abs();
                    assert!(diff < 1e-5,
                        "node={} head={} d={}: tree {:.6} vs naive {:.6}",
                        i, h, d, out[out_idx], expected_i[exp_idx]);
                }
            }
        }
    }

    #[test]
    fn masked_positions_do_not_leak_into_output() {
        // Build a tree where node 3 does NOT have node 2 as an ancestor.
        // Write huge values into the V row for position `seq_prefix + 2`.
        // Query 3's output must be insensitive to that value — if it isn't,
        // the mask is being ignored.
        let head_dim = 4;
        let n_heads = 1;
        let n_kv = 1;
        let prefix = 0;
        let scale = 1.0;
        let t = TreeSpec::full(2, 2).unwrap();  // binary tree, 7 nodes
        let mask = build_ancestor_mask(&t).unwrap();
        let n_nodes = t.n_nodes();

        let q: Vec<f32> = vec![0.25; n_nodes * n_heads * head_dim];
        let k: Vec<f32> = vec![0.1; n_kv * (prefix + n_nodes) * head_dim];
        let mut v: Vec<f32> = vec![1.0; n_kv * (prefix + n_nodes) * head_dim];
        let out_a = tree_attention_cpu(
            &q, &k, &v, &mask, n_nodes, n_heads, n_kv, head_dim, prefix, scale,
        ).unwrap();

        // Now poison node 2's V row with a large value. Any query that
        // doesn't have bit 2 set in its mask must be unchanged.
        let bad_pos = prefix + 2;
        for d in 0..head_dim {
            v[bad_pos * head_dim + d] = 9999.0;
        }
        let out_b = tree_attention_cpu(
            &q, &k, &v, &mask, n_nodes, n_heads, n_kv, head_dim, prefix, scale,
        ).unwrap();

        // Node 3: mask is 0b001011 — bit 2 NOT set → unchanged.
        // Node 4: mask is 0b010011 — bit 2 NOT set → unchanged.
        // Node 2: mask includes bit 2 → MUST change.
        // Node 5, 6: mask includes bit 2 → MUST change.
        let unchanged_nodes = [3usize, 4];
        let changed_nodes   = [2usize, 5, 6];
        for &n in &unchanged_nodes {
            let off = n * n_heads * head_dim;
            for d in 0..n_heads * head_dim {
                let diff = (out_a[off + d] - out_b[off + d]).abs();
                assert!(diff < 1e-5,
                    "node {} changed despite mask: {} vs {}",
                    n, out_a[off + d], out_b[off + d]);
            }
        }
        for &n in &changed_nodes {
            let off = n * n_heads * head_dim;
            let mut max_diff = 0f32;
            for d in 0..n_heads * head_dim {
                max_diff = max_diff.max((out_a[off + d] - out_b[off + d]).abs());
            }
            assert!(max_diff > 1.0,
                "node {} should reflect poisoned V but diff is {}",
                n, max_diff);
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    /// Plain-vanilla single-step attention over the entire K/V range.
    /// Used as the ground truth for the degenerate-tree tests.
    fn naive_attention(
        q: &[f32], k: &[f32], v: &[f32],
        n_heads: usize, n_kv: usize, head_dim: usize,
        seq_kv: usize, scale: f32,
    ) -> Vec<f32> {
        let group = n_heads / n_kv;
        let n_q = q.len() / (n_heads * head_dim);
        let mut out = vec![0f32; n_q * n_heads * head_dim];
        for i in 0..n_q {
            for h in 0..n_heads {
                let kv_h = h / group;
                let q_off = (i * n_heads + h) * head_dim;
                let q_row = &q[q_off .. q_off + head_dim];
                let mut scores = vec![0f32; seq_kv];
                let mut max_s = f32::NEG_INFINITY;
                for p in 0..seq_kv {
                    let k_off = (kv_h * seq_kv + p) * head_dim;
                    let mut s = 0f32;
                    for d in 0..head_dim {
                        s += q_row[d] * k[k_off + d];
                    }
                    s *= scale;
                    scores[p] = s;
                    if s > max_s { max_s = s; }
                }
                let mut denom = 0f32;
                for p in 0..seq_kv {
                    scores[p] = (scores[p] - max_s).exp();
                    denom += scores[p];
                }
                for p in 0..seq_kv {
                    let w = scores[p] / denom;
                    let v_off = (kv_h * seq_kv + p) * head_dim;
                    for d in 0..head_dim {
                        out[q_off + d] += w * v[v_off + d];
                    }
                }
            }
        }
        out
    }

    /// Reshape a K/V tensor `[n_kv, seq_full, hd]` into `[n_kv, seq_short, hd]`
    /// by taking the first `seq_short` positions per head.
    fn truncate_kv(src: &[f32], n_kv: usize, seq_full: usize, hd: usize,
                   seq_short: usize) -> Vec<f32> {
        let mut out = vec![0f32; n_kv * seq_short * hd];
        for h in 0..n_kv {
            for p in 0..seq_short {
                let src_off = (h * seq_full + p) * hd;
                let dst_off = (h * seq_short + p) * hd;
                out[dst_off .. dst_off + hd]
                    .copy_from_slice(&src[src_off .. src_off + hd]);
            }
        }
        out
    }
}
