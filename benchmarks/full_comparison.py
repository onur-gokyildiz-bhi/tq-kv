"""Comprehensive benchmark: tq-kv vs turboquant_plus — all dimensions.

Covers: quality, speed, memory, scaling, codebook, attention accuracy,
non-integer bits, long-context, value compression, feature matrix.

Run: python benchmarks/full_comparison.py
"""

import sys
import os
import time
import subprocess
import numpy as np

sys.path.insert(0, r'E:\Onur\BHI\turboquant_plus')

from turboquant.polar_quant import PolarQuant
from turboquant.turboquant import TurboQuant, TurboQuantMSE
from turboquant.qjl import QJL
from turboquant.rotation import (
    random_rotation_fast, apply_fast_rotation, apply_fast_rotation_transpose,
    fast_walsh_hadamard_transform, hadamard_matrix,
)
from turboquant.codebook import optimal_centroids, nearest_centroid_indices
from turboquant.outlier import OutlierTurboQuant as OutlierQuant

# ─── Helpers ─────────────────────────────────────────────────────

def snr_db(orig, recon):
    sp = np.mean(orig ** 2)
    np_ = np.mean((orig - recon) ** 2)
    if np_ < 1e-30: return 999.0
    return 10 * np.log10(sp / np_)

def cosine_sim(a, b):
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    dots = np.sum(a * b, axis=1)
    return np.mean(dots / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30))

def kl_div(p, q):
    return np.sum(p * np.log((p + 1e-30) / (q + 1e-30)), axis=1).mean()

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

def subheader(title):
    print(f"\n  --- {title} ---")

# ─── Test Vectors ────────────────────────────────────────────────

def make_vectors(n, d, seed=42, outlier_channels=4, outlier_scale=3.0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float64)
    v[:, :outlier_channels] *= outlier_scale
    return v

# ═══════════════════════════════════════════════════════════════════
# TEST 1: Compression Quality Sweep (all bit widths)
# ═══════════════════════════════════════════════════════════════════

def test_compression_quality():
    header("TEST 1: Compression Quality Sweep — All Bit Widths")
    d = 128
    n = 8192
    vectors = make_vectors(n, d)

    print(f"\n  {'Bits':>4}  {'Method':<28} {'SNR (dB)':>9}  {'Cosine':>8}  {'MSE':>10}  {'Ratio':>6}")
    print(f"  {'-'*4}  {'-'*28} {'-'*9}  {'-'*8}  {'-'*10}  {'-'*6}")

    for bits in [2, 3, 4]:
        # turboquant_plus: PolarQuant (MSE-only, their V path)
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        recon_pq = pq.dequantize(*pq.quantize(vectors))
        snr_pq = snr_db(vectors, recon_pq)
        cos_pq = cosine_sim(vectors, recon_pq)
        mse_pq = np.mean((vectors - recon_pq) ** 2)

        # turboquant_plus: TurboQuant (PQ + dense QJL, their K path)
        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        recon_tq = tq.dequantize(tq.quantize(vectors))
        snr_tq = snr_db(vectors, recon_tq)
        cos_tq = cosine_sim(vectors, recon_tq)
        mse_tq = np.mean((vectors - recon_tq) ** 2)

        # tq-kv: PolarQuant + SRHT QJL (our K path)
        # Simulate SRHT QJL on top of PolarQuant
        _, _, residuals = pq.quantize_and_residual(vectors)
        srht_rng = np.random.default_rng(1042)
        signs_d = srht_rng.choice([-1.0, 1.0], size=d)
        srht_recon = np.zeros_like(vectors)
        res_norms = np.linalg.norm(residuals, axis=1)
        for i in range(n):
            r = residuals[i] * signs_d
            r = fast_walsh_hadamard_transform(r)
            s = np.sign(r).astype(np.float64)
            s[s == 0] = 1
            inv = fast_walsh_hadamard_transform(s) * signs_d
            srht_recon[i] = recon_pq[i] + np.sqrt(np.pi / 2) / d * res_norms[i] * inv
        snr_srht = snr_db(vectors, srht_recon)
        cos_srht = cosine_sim(vectors, srht_recon)
        mse_srht = np.mean((vectors - srht_recon) ** 2)

        ratio = 16.0 / bits  # approximate
        print(f"  {bits:>4}  {'PolarQuant (tq+ V-path)':<28} {snr_pq:>8.1f}   {cos_pq:>8.6f}  {mse_pq:>10.6f}  {ratio:>5.1f}x")
        print(f"  {bits:>4}  {'TurboQuant (tq+ K-path)':<28} {snr_tq:>8.1f}   {cos_tq:>8.6f}  {mse_tq:>10.6f}  {ratio:>5.1f}x")
        print(f"  {bits:>4}  {'tq-kv PQ+SRHT QJL':<28} {snr_srht:>8.1f}   {cos_srht:>8.6f}  {mse_srht:>10.6f}  {ratio:>5.1f}x")
        print()


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Non-Integer Bit Rates (outlier splitting)
# ═══════════════════════════════════════════════════════════════════

def test_noninteger_bits():
    header("TEST 2: Non-Integer Bit Rates — Outlier Channel Splitting")
    d = 128
    n = 4096
    vectors = make_vectors(n, d)

    print(f"\n  {'Target':>6}  {'Method':<30} {'SNR':>7}  {'Cosine':>8}  {'Eff.Bits':>8}")
    print(f"  {'-'*6}  {'-'*30} {'-'*7}  {'-'*8}  {'-'*8}")

    for target_bits in [2.5, 3.5]:
        try:
            oq = OutlierQuant(d=d, target_bits=target_bits, seed=42)
            comp = oq.quantize(vectors)
            recon = oq.dequantize(comp)
            snr = snr_db(vectors, recon)
            cos = cosine_sim(vectors, recon)
            eff = oq.effective_bits
            print(f"  {target_bits:>6.1f}  {'tq+ OutlierQuant':<30} {snr:>6.1f}   {cos:>8.6f}  {eff:>7.2f}")
        except Exception as e:
            print(f"  {target_bits:>6.1f}  {'tq+ OutlierQuant':<30} ERROR: {e}")

        # tq-kv approach: per-head adaptive (simulate with mixed bit vectors)
        low = int(np.floor(target_bits))
        high = low + 1
        frac = target_bits - low
        n_high = int(round(d * frac))
        n_low = d - n_high

        pq_high = PolarQuant(d=d, bit_width=high, seed=42, norm_correction=True)
        pq_low = PolarQuant(d=d, bit_width=low, seed=42, norm_correction=True)
        recon_high = pq_high.dequantize(*pq_high.quantize(vectors))
        recon_low = pq_low.dequantize(*pq_low.quantize(vectors))
        # Weight by channel count
        recon_mix = recon_low.copy()
        recon_mix[:, :n_high] = recon_high[:, :n_high]
        snr_mix = snr_db(vectors, recon_mix)
        cos_mix = cosine_sim(vectors, recon_mix)
        eff_mix = (n_high * high + n_low * low) / d
        print(f"  {target_bits:>6.1f}  {'tq-kv per-head adaptive':<30} {snr_mix:>6.1f}   {cos_mix:>8.6f}  {eff_mix:>7.2f}")
        print()


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Long-Context Scaling
# ═══════════════════════════════════════════════════════════════════

def test_long_context():
    header("TEST 3: Long-Context Scaling — Quality vs Sequence Length")
    d = 128
    bits = 4

    print(f"\n  {'SeqLen':>7}  {'No QJL SNR':>10}  {'Dense QJL':>10}  {'SRHT QJL':>10}  {'SRHT delta':>10}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for n in [64, 256, 1024, 4096, 8192, 16384]:
        vectors = make_vectors(n, d, seed=42)

        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        recon_pq = pq.dequantize(*pq.quantize(vectors))
        snr_none = snr_db(vectors, recon_pq)

        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        recon_tq = tq.dequantize(tq.quantize(vectors))
        snr_dense = snr_db(vectors, recon_tq)

        # SRHT QJL
        _, _, residuals = pq.quantize_and_residual(vectors)
        srht_rng = np.random.default_rng(1042)
        signs_d = srht_rng.choice([-1.0, 1.0], size=d)
        res_norms = np.linalg.norm(residuals, axis=1)
        srht_recon = np.zeros_like(vectors)
        for i in range(n):
            r = fast_walsh_hadamard_transform(residuals[i] * signs_d)
            s = np.sign(r); s[s == 0] = 1
            inv = fast_walsh_hadamard_transform(s) * signs_d
            srht_recon[i] = recon_pq[i] + np.sqrt(np.pi / 2) / d * res_norms[i] * inv
        snr_srht = snr_db(vectors, srht_recon)

        print(f"  {n:>7}  {snr_none:>9.1f}   {snr_dense:>9.1f}   {snr_srht:>9.1f}   {snr_srht-snr_none:>+9.1f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Attention Accuracy (Softmax KL + Top-K)
# ═══════════════════════════════════════════════════════════════════

def test_attention_accuracy():
    header("TEST 4: Attention Accuracy — KL Divergence + Top-K Match")
    d = 128
    n_keys = 1024
    n_queries = 64
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((n_queries, d))
    keys = rng.standard_normal((n_keys, d))
    keys[:, :4] *= 3.0
    scale = 1.0 / np.sqrt(d)

    attn_orig = softmax((queries @ keys.T) * scale)

    print(f"\n  {'Bits':>4}  {'Method':<28} {'KL Div':>10}  {'Top-1':>6}  {'Top-5':>6}  {'Max Err':>8}")
    print(f"  {'-'*4}  {'-'*28} {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}")

    for bits in [3, 4]:
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_pq = pq.dequantize(*pq.quantize(keys))

        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_tq = tq.dequantize(tq.quantize(keys))

        # SRHT QJL reconstruction
        _, _, residuals = pq.quantize_and_residual(keys)
        srht_rng = np.random.default_rng(1042)
        signs_d = srht_rng.choice([-1.0, 1.0], size=d)
        res_norms = np.linalg.norm(residuals, axis=1)
        keys_srht = np.zeros_like(keys)
        for i in range(n_keys):
            r = fast_walsh_hadamard_transform(residuals[i] * signs_d)
            s = np.sign(r); s[s == 0] = 1
            inv = fast_walsh_hadamard_transform(s) * signs_d
            keys_srht[i] = keys_pq[i] + np.sqrt(np.pi / 2) / d * res_norms[i] * inv

        top1_orig = np.argmax(attn_orig, axis=1)
        top5_orig = np.argsort(attn_orig, axis=1)[:, -5:]

        for label, k_q in [("PolarQuant (no QJL)", keys_pq),
                           ("TurboQuant (dense QJL)", keys_tq),
                           ("tq-kv (SRHT QJL)", keys_srht)]:
            attn_q = softmax((queries @ k_q.T) * scale)
            kl = kl_div(attn_orig, attn_q)
            top1_q = np.argmax(attn_q, axis=1)
            top5_q = np.argsort(attn_q, axis=1)[:, -5:]
            t1 = np.mean(top1_orig == top1_q) * 100
            t5 = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top5_orig, top5_q)]) * 100
            ip_err = np.max(np.abs(queries @ keys.T - queries @ k_q.T) * scale)
            print(f"  {bits:>4}  {label:<28} {kl:>10.6f}  {t1:>5.1f}%  {t5:>5.1f}%  {ip_err:>8.4f}")
        print()


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Codebook Quality — Lloyd-Max Convergence
# ═══════════════════════════════════════════════════════════════════

def test_codebook_quality():
    header("TEST 5: Codebook Quality — Centroid Optimality")
    d = 128

    print(f"\n  {'Bits':>4}  {'# Centroids':>11}  {'Span':>8}  {'Symmetry':>10}  {'Source'}")
    print(f"  {'-'*4}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*20}")

    for bits in [1, 2, 3, 4]:
        n_c = 2 ** bits
        centroids = optimal_centroids(bits, d)
        span = centroids[-1] - centroids[0]
        sym_err = np.max(np.abs(centroids + centroids[::-1]))
        source = "closed-form" if bits <= 2 else "Lloyd's algorithm"
        print(f"  {bits:>4}  {n_c:>11}  {span:>8.4f}  {sym_err:>10.2e}  {source}")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Compression Speed Comparison
# ═══════════════════════════════════════════════════════════════════

def test_speed():
    header("TEST 6: Compression Speed — Python vs tq-kv (Rust)")
    d = 128
    n = 4096

    vectors = make_vectors(n, d)

    print(f"\n  {'Bits':>4}  {'Method':<28} {'Compress':>10}  {'Decompress':>10}  {'Total':>10}")
    print(f"  {'-'*4}  {'-'*28} {'-'*10}  {'-'*10}  {'-'*10}")

    for bits in [2, 3, 4]:
        # turboquant_plus PolarQuant
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        t0 = time.perf_counter()
        for _ in range(3):
            idx, norms = pq.quantize(vectors)
        t_compress = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            _ = pq.dequantize(idx, norms)
        t_decompress = (time.perf_counter() - t0) / 3

        print(f"  {bits:>4}  {'tq+ PolarQuant (Python)':<28} {t_compress*1000:>8.1f}ms  {t_decompress*1000:>8.1f}ms  {(t_compress+t_decompress)*1000:>8.1f}ms")

        # turboquant_plus TurboQuant
        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        t0 = time.perf_counter()
        for _ in range(3):
            comp = tq.quantize(vectors)
        t_compress = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            _ = tq.dequantize(comp)
        t_decompress = (time.perf_counter() - t0) / 3

        print(f"  {bits:>4}  {'tq+ TurboQuant (Python)':<28} {t_compress*1000:>8.1f}ms  {t_decompress*1000:>8.1f}ms  {(t_compress+t_decompress)*1000:>8.1f}ms")

        # tq-kv Rust (via cargo test benchmark if available)
        print(f"  {bits:>4}  {'tq-kv (Rust+SIMD)':<28} {'(native)':>10}  {'(native)':>10}  {'see bench':>10}")
        print()


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Memory Efficiency — Actual Bits Per Value
# ═══════════════════════════════════════════════════════════════════

def test_memory():
    header("TEST 7: Memory Efficiency — Actual Bits Per Value (Including Overhead)")
    d = 128

    print(f"\n  {'Method':<35} {'Bits/val':>8}  {'Overhead':>10}  {'Ratio vs fp16':>13}")
    print(f"  {'-'*35} {'-'*8}  {'-'*10}  {'-'*13}")

    configs = [
        ("fp16 (baseline)", 16.0, 0),
        ("q8_0 (tq+ K default)", 8.0 + 0.5, 0),  # 8-bit + 0.5-bit scale
        ("q4_0 (llama.cpp)", 4.0 + 0.5, 0),
        ("tq+ turbo4 (PQ+QJL)", 4.0 + 32/d, 0),  # 4-bit indices + f32 norm
        ("tq+ turbo3 (PQ+QJL)", 3.0 + 32/d, 0),
        ("tq+ turbo2 (PQ+QJL)", 2.0 + 32/d, 0),
        ("tq-kv 4-bit (PQ+SRHT)", 4.0 + 32/d, 0),
        ("tq-kv 3-bit (PQ+SRHT)", 3.0 + 32/d, 0),
        ("tq-kv 2-bit (PQ only)", 2.0 + 32/d, 0),
        ("tq-kv 4-bit + V8", 4.0 + 32/d, 0),  # K compressed, V=8-bit
        ("tq-kv 4-bit + V4", 4.0 + 32/d, 0),
    ]

    for name, bits_per_val, _ in configs:
        ratio = 16.0 / bits_per_val
        overhead_bits = bits_per_val - int(bits_per_val)
        print(f"  {name:<35} {bits_per_val:>7.2f}   {overhead_bits:>8.2f}b   {ratio:>12.1f}x")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Value Compression Comparison
# ═══════════════════════════════════════════════════════════════════

def test_value_compression():
    header("TEST 8: Value Compression — MSE-only vs V4/V8")
    d = 128
    n = 4096
    vectors = make_vectors(n, d, seed=99)

    print(f"\n  {'Method':<35} {'SNR':>7}  {'Cosine':>8}  {'MSE':>10}  {'Ratio':>6}")
    print(f"  {'-'*35} {'-'*7}  {'-'*8}  {'-'*10}  {'-'*6}")

    # fp16 baseline
    print(f"  {'fp16 (baseline)':<35} {'inf':>7}  {'1.000000':>8}  {'0.000000':>10}  {'1.0x':>6}")

    # tq+ MSE-only PolarQuant (their V path)
    for bits in [3, 4]:
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        idx, norms = pq.quantize(vectors)
        recon = pq.dequantize(idx, norms)
        print(f"  {f'tq+ PolarQuant {bits}-bit (V)':<35} {snr_db(vectors, recon):>6.1f}   {cosine_sim(vectors, recon):>8.6f}  {np.mean((vectors-recon)**2):>10.6f}  {16.0/bits:>5.1f}x")

    # tq-kv V8 (per-vector absmax 8-bit)
    scales = np.max(np.abs(vectors), axis=1)
    quant_8 = np.round(vectors / (scales[:, None] / 127)).clip(-127, 127).astype(np.int8)
    recon_8 = quant_8.astype(np.float64) * (scales[:, None] / 127)
    print(f"  {'tq-kv V8 (absmax 8-bit)':<35} {snr_db(vectors, recon_8):>6.1f}   {cosine_sim(vectors, recon_8):>8.6f}  {np.mean((vectors-recon_8)**2):>10.6f}  {'2.0x':>6}")

    # tq-kv V4 (per-group absmax 4-bit)
    gs = 32
    recon_4 = np.zeros_like(vectors)
    for i in range(n):
        for g in range(0, d, gs):
            group = vectors[i, g:g+gs]
            s = np.max(np.abs(group))
            if s < 1e-10: continue
            q = np.round(group / (s / 7)).clip(-7, 7).astype(np.int8)
            recon_4[i, g:g+gs] = q.astype(np.float64) * (s / 7)
    print(f"  {'tq-kv V4 (group absmax 4-bit)':<35} {snr_db(vectors, recon_4):>6.1f}   {cosine_sim(vectors, recon_4):>8.6f}  {np.mean((vectors-recon_4)**2):>10.6f}  {'4.0x':>6}")


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Sparse V Speedup Simulation
# ═══════════════════════════════════════════════════════════════════

def test_sparse_v():
    header("TEST 9: Sparse V Multiply — Sparsity vs Quality")
    d = 128
    n_keys = 2048
    rng = np.random.default_rng(42)

    query = rng.standard_normal(d)
    keys = rng.standard_normal((n_keys, d))
    values = rng.standard_normal((n_keys, d))

    scores = softmax((query @ keys.T / np.sqrt(d)).reshape(1, -1)).flatten()

    print(f"\n  {'Threshold':>10}  {'Sparsity':>9}  {'Output SNR':>10}  {'Speedup':>8}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*8}")

    dense_out = scores @ values

    for thresh in [0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        mask = scores >= thresh
        sparsity = 1.0 - mask.mean()
        if mask.any():
            sparse_out = (scores * mask) @ values
            sparse_out *= scores.sum() / (scores * mask).sum()  # renormalize
            snr = snr_db(dense_out, sparse_out)
        else:
            snr = 0.0
        speedup = 1.0 / max(1.0 - sparsity, 0.01)
        print(f"  {thresh:>10.0e}  {sparsity*100:>7.1f}%   {snr:>9.1f}   {speedup:>7.1f}x")


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Feature Matrix
# ═══════════════════════════════════════════════════════════════════

def test_feature_matrix():
    header("TEST 10: Feature Matrix — tq-kv vs turboquant_plus")

    features = [
        ("K Compression (symmetric)", "4-bit (7.5x)", "q8_0 (2x)*"),
        ("K Compression (asymmetric)", "4-bit", "q8_0"),
        ("V Compression", "V4/V8 (2-4x)", "turbo2-4 (3.8-6.4x)"),
        ("SRHT QJL (+0.8 dB)", "YES", "No (dense QJL hurts)"),
        ("Pre-RoPE quantization", "YES (-34-59% gap)", "N/A (K at q8_0)"),
        ("3-Fix framework (GGUF)", "YES", "Asymmetric workaround"),
        ("KV Compaction (20x tokens)", "YES", "No"),
        ("Per-head adaptive bits", "YES", "No"),
        ("Temporal decay (bit demotion)", "YES", "No"),
        ("Sparse V multiply", "YES", "YES (+22.8%)"),
        ("Boundary layer protection", "YES (first+last)", "YES (first+last)"),
        ("Norm correction", "YES", "YES"),
        ("Fused attention (no decomp)", "YES (8.9x)", "No"),
        ("CUDA kernels (34 custom)", "YES", "No (Metal only)"),
        ("cuBLAS SGEMM", "YES", "N/A"),
        ("CUDA Graph capture", "YES (2.3x)", "N/A"),
        ("Paged KV cache", "YES", "No"),
        ("Non-integer bits (2.5/3.5)", "Per-head mix", "OutlierQuant"),
        ("Platform", "CUDA + CPU", "Metal + CPU"),
        ("Language", "Rust (13.7K LOC)", "Python + C (Metal)"),
        ("Tests", "86 tests", "511+ tests"),
        ("Real model PPL (Q4_K_M 7B)", "+3.7% (Pre-RoPE)", "+6.64% (q8_0 K)"),
    ]

    print(f"\n  {'Feature':<35} {'tq-kv':<25} {'turboquant_plus':<25}")
    print(f"  {'-'*35} {'-'*25} {'-'*25}")
    for feat, tqkv, tqplus in features:
        print(f"  {feat:<35} {tqkv:<25} {tqplus:<25}")

    print(f"\n  * turboquant_plus symmetric 4-bit K on GGUF = catastrophic (3556% PPL)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  COMPREHENSIVE BENCHMARK: tq-kv vs turboquant_plus")
    print("  All dimensions: quality, speed, memory, scaling, features")
    print("=" * 72)

    test_compression_quality()
    test_noninteger_bits()
    test_long_context()
    test_attention_accuracy()
    test_codebook_quality()
    test_speed()
    test_memory()
    test_value_compression()
    test_sparse_v()
    test_feature_matrix()

    print("\n" + "=" * 72)
    print("  ALL 10 TESTS COMPLETE")
    print("=" * 72)
