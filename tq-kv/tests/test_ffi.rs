//! FFI integration test — validates the C API works correctly.
//! This tests the same functions that C code would call via tq_kv.h.

#[cfg(feature = "ffi")]
mod ffi_tests {
    use tq_kv::ffi::*;
    use std::ptr;

    const HEAD_DIM: u32 = 128;

    fn make_data(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|i| (seed * (i as f32 + 1.0) * 0.1).sin() * 0.5).collect()
    }

    #[test]
    fn test_single_head_lifecycle() {
        let ctx = tq_init(2, HEAD_DIM, 0);
        assert!(!ctx.is_null());

        // Append 64 keys
        for k in 0..64 {
            let key = make_data(HEAD_DIM as usize, (k + 1) as f32);
            let rc = unsafe { tq_compress_and_append(ctx, key.as_ptr(), HEAD_DIM) };
            assert_eq!(rc, 0);
        }
        assert_eq!(unsafe { tq_cached_count(ctx) }, 64);

        // Check compression ratio
        let compressed = unsafe { tq_memory_bytes(ctx) };
        let original = unsafe { tq_original_memory_bytes(ctx) };
        let ratio = original as f64 / compressed as f64;
        assert!(ratio > 5.0, "2-bit ratio should be > 5x, got {:.1}", ratio);

        // Fused attention
        let query = make_data(HEAD_DIM as usize, 99.0);
        let mut scores = vec![0.0f32; 64];
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let n = unsafe {
            tq_fused_attention(ctx, query.as_ptr(), HEAD_DIM, scores.as_mut_ptr(), scale)
        };
        assert_eq!(n, 64);
        assert!(scores.iter().all(|s| s.is_finite()));
        assert!(scores.iter().any(|s| s.abs() > 1e-10));

        // Clear + free
        unsafe { tq_clear(ctx) };
        assert_eq!(unsafe { tq_cached_count(ctx) }, 0);
        unsafe { tq_free(ctx) };
    }

    #[test]
    fn test_layer_api() {
        let n_heads: u32 = 8;
        let ctx = tq_layer_init(4, n_heads, HEAD_DIM, 0);
        assert!(!ctx.is_null());

        // Append 32 tokens
        for t in 0..32 {
            let keys = make_data((n_heads * HEAD_DIM) as usize, (t + 1) as f32);
            let rc = unsafe {
                tq_layer_compress_and_append(ctx, keys.as_ptr(), n_heads * HEAD_DIM)
            };
            assert_eq!(rc, 0);
        }
        assert_eq!(unsafe { tq_layer_cached_count(ctx) }, 32);

        // Fused attention per head
        let query = make_data(HEAD_DIM as usize, 42.0);
        let mut scores = vec![0.0f32; 32];
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        for h in 0..n_heads {
            let n = unsafe {
                tq_layer_fused_attention(ctx, h, query.as_ptr(), HEAD_DIM, scores.as_mut_ptr(), scale)
            };
            assert_eq!(n, 32);
        }

        // Memory
        let mem = unsafe { tq_layer_memory_bytes(ctx) };
        assert!(mem > 0);

        unsafe { tq_layer_clear(ctx) };
        assert_eq!(unsafe { tq_layer_cached_count(ctx) }, 0);
        unsafe { tq_layer_free(ctx) };
    }

    #[test]
    fn test_null_safety() {
        // All functions should handle null gracefully
        assert_eq!(unsafe { tq_compress_and_append(ptr::null_mut(), ptr::null(), 0) }, -1);
        assert_eq!(unsafe { tq_fused_attention(ptr::null(), ptr::null(), 0, ptr::null_mut(), 0.0) }, -1);
        assert_eq!(unsafe { tq_cached_count(ptr::null()) }, 0);
        unsafe { tq_free(ptr::null_mut()) }; // should not crash
        unsafe { tq_layer_free(ptr::null_mut()) }; // should not crash
    }

    #[test]
    fn test_all_bitwidths() {
        for bits in [2u8, 3, 4] {
            let ctx = tq_init(bits, HEAD_DIM, 0);
            for k in 0..16 {
                let key = make_data(HEAD_DIM as usize, (k + 1) as f32);
                unsafe { tq_compress_and_append(ctx, key.as_ptr(), HEAD_DIM) };
            }
            let compressed = unsafe { tq_memory_bytes(ctx) };
            let original = unsafe { tq_original_memory_bytes(ctx) };
            let ratio = original as f64 / compressed as f64;
            eprintln!("{}-bit: {:.1}x compression", bits, ratio);
            assert!(ratio > 2.0);
            unsafe { tq_free(ctx) };
        }
    }
}
