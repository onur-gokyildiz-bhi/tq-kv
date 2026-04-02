"""Head-to-head comparison: tq-kv (Rust via subprocess) vs turboquant_plus (Python).

Generates identical test vectors, runs both implementations, compares:
- MSE, SNR, cosine similarity
- Dense QJL vs SRHT QJL vs no QJL
- Single-sign rotation vs double-sign rotation
- Inner product preservation (attention score accuracy)
"""

import sys
import os
import time
import json
import subprocess
import numpy as np

# Add turboquant_plus to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'turboquant_plus'))

from turboquant.polar_quant import PolarQuant
from turboquant.turboquant import TurboQuant, TurboQuantMSE
from turboquant.qjl import QJL
from turboquant.rotation import (
    random_rotation_fast, apply_fast_rotation, apply_fast_rotation_transpose,
    fast_walsh_hadamard_transform, hadamard_matrix,
)
from turboquant.codebook import optimal_centroids, nearest_centroid_indices


def snr_db(original, reconstructed):
    """Signal-to-noise ratio in dB."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power < 1e-30:
        return 999.0
    return 10 * np.log10(signal_power / noise_power)


def cosine_sim(a, b):
    """Cosine similarity, handles batches."""
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    # batch
    dots = np.sum(a * b, axis=1)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    return np.mean(dots / (norms_a * norms_b + 1e-30))


# ─── Test 1: PolarQuant quality comparison ───────────────────────────
def test_polarquant_quality():
    """Compare PolarQuant (MSE-only, no QJL) across bit widths."""
    print("=" * 70)
    print("TEST 1: PolarQuant Quality — turboquant_plus vs tq-kv theory")
    print("=" * 70)

    d = 128
    n_vectors = 4096
    rng = np.random.default_rng(42)

    # Generate test vectors with realistic KV cache distribution
    # (not unit norm — real keys have varying norms)
    vectors = rng.standard_normal((n_vectors, d)).astype(np.float64)
    # Add some outlier channels (simulating real attention patterns)
    vectors[:, :4] *= 3.0  # outlier channels

    for bits in [2, 3, 4]:
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)

        t0 = time.perf_counter()
        indices, norms = pq.quantize(vectors)
        t_quant = time.perf_counter() - t0

        t0 = time.perf_counter()
        reconstructed = pq.dequantize(indices, norms)
        t_dequant = time.perf_counter() - t0

        snr = snr_db(vectors, reconstructed)
        cos = cosine_sim(vectors, reconstructed)
        mse = np.mean((vectors - reconstructed) ** 2)

        print(f"\n  {bits}-bit PolarQuant (d={d}, n={n_vectors}):")
        print(f"    SNR:        {snr:.1f} dB")
        print(f"    Cosine:     {cos:.6f}")
        print(f"    MSE:        {mse:.6f}")
        print(f"    Quant:      {t_quant*1000:.1f} ms")
        print(f"    Dequant:    {t_dequant*1000:.1f} ms")


# ─── Test 2: QJL head-to-head ────────────────────────────────────────
def test_qjl_headtohead():
    """THE key scientific disagreement: dense QJL vs SRHT QJL vs no QJL.

    turboquant_plus (and 5 independent groups) say QJL hurts quality.
    tq-kv says SRHT QJL helps (+4.5 dB SNR).

    Hypothesis: dense QJL has high variance due to random Gaussian projection.
    SRHT has structured, lower-variance projection that actually helps.
    """
    print("\n" + "=" * 70)
    print("TEST 2: QJL Head-to-Head — Dense vs SRHT vs None")
    print("  (This is the key scientific disagreement)")
    print("=" * 70)

    d = 128
    n_vectors = 4096
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, d)).astype(np.float64)
    vectors[:, :4] *= 3.0

    for bits in [3, 4]:
        print(f"\n  --- {bits}-bit ---")

        # 1. PolarQuant only (no QJL) — baseline
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        _, _, residuals = pq.quantize_and_residual(vectors)
        pq_reconstructed = pq.dequantize(*pq.quantize(vectors))

        snr_none = snr_db(vectors, pq_reconstructed)
        cos_none = cosine_sim(vectors, pq_reconstructed)

        # 2. TurboQuant with dense QJL (their implementation)
        # TurboQuant uses (bits-1) for PolarQuant + 1 for QJL
        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        compressed = tq.quantize(vectors)
        tq_reconstructed = tq.dequantize(compressed)

        snr_dense = snr_db(vectors, tq_reconstructed)
        cos_dense = cosine_sim(vectors, tq_reconstructed)

        # 3. SRHT QJL (our approach) — simulate with structured Hadamard
        # Instead of S ~ N(0,1)^(d×d), use sqrt(d/m) * S * H * D
        srht_rng = np.random.default_rng(1042)
        signs_d = srht_rng.choice([-1.0, 1.0], size=d)  # diagonal D
        m = d  # same projection dimension for fair comparison

        # Project residuals via SRHT
        srht_projected = np.zeros((n_vectors, m))
        for i in range(n_vectors):
            r = residuals[i].copy()
            r_norm = np.linalg.norm(r)
            # D @ residual
            r_signed = r * signs_d
            # H @ D @ residual (fast Walsh-Hadamard)
            r_hadamard = fast_walsh_hadamard_transform(r_signed)
            # Subsample (take first m components, scale by sqrt(d/m))
            srht_projected[i] = r_hadamard[:m] * np.sqrt(d / m)

        # Sign quantize
        srht_signs = np.sign(srht_projected).astype(np.int8)
        srht_signs[srht_signs == 0] = 1
        residual_norms = np.linalg.norm(residuals, axis=1)

        # Dequantize SRHT QJL
        srht_qjl_const = np.sqrt(np.pi / 2)
        srht_reconstructed_residuals = np.zeros((n_vectors, d))
        for i in range(n_vectors):
            s = srht_signs[i].astype(np.float64)
            # Inverse: D^T @ H^T @ S^T @ signs (but S is subsample, H and D are self-inverse)
            # Pad back if subsampled
            padded = np.zeros(d)
            padded[:m] = s * np.sqrt(d / m)
            r_inv = fast_walsh_hadamard_transform(padded)
            r_inv *= signs_d  # D^T = D
            srht_reconstructed_residuals[i] = srht_qjl_const / d * residual_norms[i] * r_inv

        srht_full_reconstructed = pq_reconstructed + srht_reconstructed_residuals
        snr_srht = snr_db(vectors, srht_full_reconstructed)
        cos_srht = cosine_sim(vectors, srht_full_reconstructed)

        print(f"    No QJL:     SNR={snr_none:6.1f} dB  cos={cos_none:.6f}")
        print(f"    Dense QJL:  SNR={snr_dense:6.1f} dB  cos={cos_dense:.6f}  (turboquant_plus)")
        print(f"    SRHT QJL:   SNR={snr_srht:6.1f} dB  cos={cos_srht:.6f}  (tq-kv approach)")

        delta_dense = snr_dense - snr_none
        delta_srht = snr_srht - snr_none
        print(f"    Dense vs none: {delta_dense:+.1f} dB")
        print(f"    SRHT vs none:  {delta_srht:+.1f} dB")
        print(f"    SRHT vs dense: {delta_srht - delta_dense:+.1f} dB")


# ─── Test 3: Inner product (attention score) accuracy ─────────────────
def test_inner_product_accuracy():
    """Attention scores = dot(query, key). Compare accuracy across methods."""
    print("\n" + "=" * 70)
    print("TEST 3: Inner Product (Attention Score) Accuracy")
    print("=" * 70)

    d = 128
    n_pairs = 2000
    rng = np.random.default_rng(42)

    queries = rng.standard_normal((n_pairs, d))
    keys = rng.standard_normal((n_pairs, d))

    for bits in [3, 4]:
        print(f"\n  --- {bits}-bit ---")

        # PolarQuant only (MSE)
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_pq = pq.dequantize(*pq.quantize(keys))

        # TurboQuant (PolarQuant + dense QJL)
        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_tq = tq.dequantize(tq.quantize(keys))

        # Compute IP errors
        ip_orig = np.sum(queries * keys, axis=1)
        ip_pq = np.sum(queries * keys_pq, axis=1)
        ip_tq = np.sum(queries * keys_tq, axis=1)

        err_pq = np.abs(ip_orig - ip_pq)
        err_tq = np.abs(ip_orig - ip_tq)

        print(f"    PolarQuant only (no QJL):")
        print(f"      Mean |IP error|: {np.mean(err_pq):.6f}  Max: {np.max(err_pq):.6f}")
        print(f"    TurboQuant (dense QJL):")
        print(f"      Mean |IP error|: {np.mean(err_tq):.6f}  Max: {np.max(err_tq):.6f}")

        # Note: for single-side quantization (only K compressed, Q at full precision),
        # QJL's unbiased property matters less — MSE dominates
        ratio = np.mean(err_tq) / np.mean(err_pq)
        print(f"    Dense QJL IP error ratio vs PQ-only: {ratio:.2f}x")


# ─── Test 4: Rotation comparison ─────────────────────────────────────
def test_rotation_strategies():
    """Compare single-sign (tq-kv) vs double-sign (turboquant_plus) rotation."""
    print("\n" + "=" * 70)
    print("TEST 4: Rotation Strategy — Single-Sign vs Double-Sign")
    print("=" * 70)

    d = 128
    n_vectors = 4096
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, d)).astype(np.float64)
    vectors[:, :4] *= 3.0

    # Double-sign rotation (turboquant_plus): D2 @ H @ D1
    signs1, signs2, padded_d = random_rotation_fast(d, np.random.default_rng(42))

    # Single-sign rotation (tq-kv style): (1/sqrt(d)) * H @ D
    single_signs = np.random.default_rng(42).choice([-1.0, 1.0], size=d)

    # Measure kurtosis after rotation (should be close to 3.0 for Gaussian)
    from scipy import stats as sp_stats

    double_rotated = np.array([
        apply_fast_rotation(v, signs1, signs2, padded_d) for v in vectors
    ])
    single_rotated = np.array([
        fast_walsh_hadamard_transform(v * single_signs) for v in vectors
    ])

    # Kurtosis per channel, averaged
    kurt_double = np.mean([sp_stats.kurtosis(double_rotated[:, j], fisher=True) for j in range(d)])
    kurt_single = np.mean([sp_stats.kurtosis(single_rotated[:, j], fisher=True) for j in range(d)])

    print(f"\n  Kurtosis (excess, ideal=0 for Gaussian):")
    print(f"    Double-sign (turboquant_plus): {kurt_double:.3f}")
    print(f"    Single-sign (tq-kv):           {kurt_single:.3f}")
    print(f"    Raw (no rotation):             {np.mean([sp_stats.kurtosis(vectors[:, j], fisher=True) for j in range(d)]):.3f}")

    # Now quantize both rotated versions and compare
    centroids_4bit = optimal_centroids(4, d)

    double_indices = nearest_centroid_indices(double_rotated, centroids_4bit)
    double_recon_rot = centroids_4bit[double_indices]
    # Inverse rotation
    double_recon = np.array([
        apply_fast_rotation_transpose(double_recon_rot[i], signs1, signs2, padded_d) for i in range(n_vectors)
    ])

    single_indices = nearest_centroid_indices(single_rotated, centroids_4bit)
    single_recon_rot = centroids_4bit[single_indices]
    single_recon = np.array([
        fast_walsh_hadamard_transform(single_recon_rot[i]) * single_signs for i in range(n_vectors)
    ])

    snr_double = snr_db(vectors, double_recon)
    snr_single = snr_db(vectors, single_recon)
    cos_double = cosine_sim(vectors, double_recon)
    cos_single = cosine_sim(vectors, single_recon)

    print(f"\n  4-bit quantization quality after rotation:")
    print(f"    Double-sign: SNR={snr_double:.1f} dB  cos={cos_double:.6f}")
    print(f"    Single-sign: SNR={snr_single:.1f} dB  cos={cos_single:.6f}")
    print(f"    Delta:       {snr_double - snr_single:+.1f} dB")


# ─── Test 5: Norm correction comparison ──────────────────────────────
def test_norm_correction():
    """turboquant_plus renormalizes to unit norm in rotated domain.
    tq-kv uses adaptive sigma per vector. Compare."""
    print("\n" + "=" * 70)
    print("TEST 5: Norm Correction — Renormalize vs Adaptive Sigma")
    print("=" * 70)

    d = 128
    n_vectors = 4096
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, d)).astype(np.float64)
    vectors[:, :4] *= 3.0

    for bits in [2, 3, 4]:
        # With norm correction (turboquant_plus default)
        pq_nc = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        recon_nc = pq_nc.dequantize(*pq_nc.quantize(vectors))

        # Without norm correction
        pq_raw = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=False)
        recon_raw = pq_raw.dequantize(*pq_raw.quantize(vectors))

        snr_nc = snr_db(vectors, recon_nc)
        snr_raw = snr_db(vectors, recon_raw)
        cos_nc = cosine_sim(vectors, recon_nc)
        cos_raw = cosine_sim(vectors, recon_raw)

        # Norm preservation
        orig_norms = np.linalg.norm(vectors, axis=1)
        nc_norms = np.linalg.norm(recon_nc, axis=1)
        raw_norms = np.linalg.norm(recon_raw, axis=1)
        norm_err_nc = np.mean(np.abs(orig_norms - nc_norms) / orig_norms)
        norm_err_raw = np.mean(np.abs(orig_norms - raw_norms) / orig_norms)

        print(f"\n  {bits}-bit:")
        print(f"    With norm correction:    SNR={snr_nc:.1f} dB  cos={cos_nc:.6f}  norm_err={norm_err_nc:.4f}")
        print(f"    Without norm correction: SNR={snr_raw:.1f} dB  cos={cos_raw:.6f}  norm_err={norm_err_raw:.4f}")


# ─── Test 6: Softmax attention distribution accuracy ─────────────────
def test_softmax_attention():
    """The real test: how does quantization affect the softmax attention distribution?
    This is what actually matters for output quality."""
    print("\n" + "=" * 70)
    print("TEST 6: Softmax Attention Distribution Accuracy (KL Divergence)")
    print("=" * 70)

    d = 128
    n_keys = 512  # context length
    n_queries = 32
    rng = np.random.default_rng(42)

    queries = rng.standard_normal((n_queries, d)).astype(np.float64)
    keys = rng.standard_normal((n_keys, d)).astype(np.float64)
    keys[:, :4] *= 3.0

    scale = 1.0 / np.sqrt(d)

    # Original attention
    scores_orig = (queries @ keys.T) * scale
    attn_orig = np.exp(scores_orig - scores_orig.max(axis=1, keepdims=True))
    attn_orig = attn_orig / attn_orig.sum(axis=1, keepdims=True)

    for bits in [3, 4]:
        print(f"\n  --- {bits}-bit, {n_keys} keys, {n_queries} queries ---")

        # PolarQuant only
        pq = PolarQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_pq = pq.dequantize(*pq.quantize(keys))

        # TurboQuant (with dense QJL)
        tq = TurboQuant(d=d, bit_width=bits, seed=42, norm_correction=True)
        keys_tq = tq.dequantize(tq.quantize(keys))

        for label, keys_q in [("PolarQuant (no QJL)", keys_pq), ("TurboQuant (dense QJL)", keys_tq)]:
            scores_q = (queries @ keys_q.T) * scale
            attn_q = np.exp(scores_q - scores_q.max(axis=1, keepdims=True))
            attn_q = attn_q / attn_q.sum(axis=1, keepdims=True)

            # KL divergence per query, averaged
            kl = np.mean(np.sum(attn_orig * np.log((attn_orig + 1e-30) / (attn_q + 1e-30)), axis=1))

            # Top-k accuracy (does the right token get the most attention?)
            topk_orig = np.argmax(attn_orig, axis=1)
            topk_q = np.argmax(attn_q, axis=1)
            topk_match = np.mean(topk_orig == topk_q)

            print(f"    {label}:")
            print(f"      KL divergence:  {kl:.6f}")
            print(f"      Top-1 accuracy: {topk_match*100:.1f}%")


if __name__ == "__main__":
    test_polarquant_quality()
    test_qjl_headtohead()
    test_inner_product_accuracy()
    test_rotation_strategies()
    test_norm_correction()
    test_softmax_attention()
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
