#!/usr/bin/env python3
"""
EAGLE draft forward — pure numpy + safetensors reference implementation.

Purpose: lock in the ground-truth value of every intermediate tensor
produced by ONE draft forward step on a FIXED deterministic input.
Dumps each intermediate to /tmp/ref_<stage>.npy so the Rust impl can
be compared element-wise and the first divergent stage pinpointed.

Dependencies: numpy + safetensors  (no torch / transformers needed).

Usage:
    python scripts/eagle_ref_numpy.py \\
        ~/.cache/huggingface/hub/eagle-qwen2-7b/model.safetensors

The fixed input is the SAME as Rust's eagle-probe `--smoke`:
    prev_hidden = zeros(3584)
    token_id    = 42
    position    = 0
"""
import sys
import pathlib
import json
import struct
import numpy as np

# ── Config (Qwen2 7B draft — mirrors src/eagle/mod.rs eagle_qwen2_7b_config) ──
HIDDEN          = 3584
INTERMEDIATE    = 18944
N_HEADS         = 28
N_KV_HEADS      = 28           # draft is full-rank (no GQA)
HEAD_DIM        = HIDDEN // N_HEADS        # 128
ROPE_DIM        = HEAD_DIM                  # Qwen2 rotates full head
ROPE_THETA      = 1_000_000.0
RMS_NORM_EPS    = 1e-6
MAX_SEQ         = 2048
VOCAB           = 152064

# ── Fixed input (matches Rust smoke) ──
PREV_HIDDEN     = np.zeros(HIDDEN, dtype=np.float32)
TOKEN_ID        = 42
POSITION        = 0


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """BF16 is stored as 2-byte uint16; upper 16 bits of FP32 mantissa+exp."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view("<f4").copy()


def load_draft_weights_raw(path: pathlib.Path):
    """Read the safetensors file manually so we can handle BF16 without torch.

    safetensors layout:
        u64 header_len (LE) | JSON header (utf-8) | raw tensor bytes
    Each JSON entry has dtype + shape + data_offsets (start, end) relative
    to the start of the raw tensor bytes region.
    """
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        f.seek(data_start)
        raw = f.read()

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        s, e = meta["data_offsets"]
        blob = raw[s:e]
        if dtype == "BF16":
            arr = bf16_bytes_to_f32(blob).reshape(shape)
        elif dtype == "F16":
            arr = np.frombuffer(blob, dtype="<f2").astype(np.float32).reshape(shape)
        elif dtype == "F32":
            arr = np.frombuffer(blob, dtype="<f4").copy().reshape(shape)
        else:
            raise RuntimeError(f"unsupported dtype {dtype} for {name}")
        out[name] = arr
    return out


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def build_rope(max_seq: int, rope_dim: int, theta: float):
    half = rope_dim // 2
    positions = np.arange(max_seq, dtype=np.float64)[:, None]
    freqs = theta ** (-2.0 * np.arange(half, dtype=np.float64) / rope_dim)
    angles = positions * freqs   # [max_seq, half]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_halved(x_h: np.ndarray, cos: np.ndarray, sin: np.ndarray, pos: int) -> np.ndarray:
    """x_h shape [n_heads, head_dim]. Qwen2 halved-RoPE: first half / second half split."""
    half = x_h.shape[-1] // 2
    x1 = x_h[:, :half]
    x2 = x_h[:, half:]
    c = cos[pos]  # [half]
    s = sin[pos]
    out = np.empty_like(x_h)
    out[:, :half] = x1 * c - x2 * s
    out[:, half:] = x1 * s + x2 * c
    return out


def dump(name: str, arr: np.ndarray, out_dir: pathlib.Path):
    p = out_dir / f"ref_{name}.npy"
    np.save(p, arr)
    l2 = float(np.linalg.norm(arr))
    mx = float(np.max(np.abs(arr)))
    head = arr.flatten()[:5].tolist()
    print(f"  [ref] {name:<18} shape={str(list(arr.shape)):<20} "
          f"|L2|={l2:.3f} |max|={mx:.3f} head[:5]={head}")


def main():
    if len(sys.argv) < 2:
        print("usage: eagle_ref_numpy.py <path/to/draft.safetensors>", file=sys.stderr)
        sys.exit(1)

    path = pathlib.Path(sys.argv[1]).expanduser()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = pathlib.Path("/tmp")
    print(f"[ref] loading {path}")
    w = load_draft_weights_raw(path)
    for k in sorted(w.keys()):
        print(f"  {k:<55} {w[k].shape}  {w[k].dtype}")

    # Resolve by expected name
    embed      = w["embed_tokens.weight"]                              # [V, H]
    fc_w       = w["fc.weight"]                                        # [H, 2H]
    fc_b       = w["fc.bias"]                                          # [H]
    q_w        = w["layers.0.self_attn.q_proj.weight"]                 # [H, H]
    q_b        = w["layers.0.self_attn.q_proj.bias"]
    k_w        = w["layers.0.self_attn.k_proj.weight"]
    k_b        = w["layers.0.self_attn.k_proj.bias"]
    v_w        = w["layers.0.self_attn.v_proj.weight"]
    v_b        = w["layers.0.self_attn.v_proj.bias"]
    o_w        = w["layers.0.self_attn.o_proj.weight"]
    gate_w     = w["layers.0.mlp.gate_proj.weight"]                    # [I, H]
    up_w       = w["layers.0.mlp.up_proj.weight"]
    down_w     = w["layers.0.mlp.down_proj.weight"]                    # [H, I]
    post_norm  = w["layers.0.post_attention_layernorm.weight"]         # [H]

    print(f"\n[ref] input: prev_hidden=zeros({HIDDEN}), token_id={TOKEN_ID}, pos={POSITION}")

    # Step 1: concat [embed(token_id), prev_hidden]
    # SafeAILab cnets1.py: torch.cat((inputs_embeds, hidden_states), dim=-1) — EMBED FIRST.
    concat = np.concatenate([embed[TOKEN_ID].astype(np.float32), PREV_HIDDEN])
    dump("01_concat", concat, out_dir)

    # Step 2: fc.weight @ concat + fc.bias
    fc_out = fc_w @ concat + fc_b
    dump("02_fc_out", fc_out, out_dir)

    # Step 3: q/k/v projections
    q = q_w @ fc_out + q_b
    k = k_w @ fc_out + k_b
    v = v_w @ fc_out + v_b
    dump("03_q", q, out_dir)
    dump("04_k", k, out_dir)
    dump("05_v", v, out_dir)

    # Step 4: RoPE on q, k at position 0 (no rotation)
    cos, sin = build_rope(MAX_SEQ, ROPE_DIM, ROPE_THETA)
    q_rope = apply_rope_halved(q.reshape(N_HEADS, HEAD_DIM), cos, sin, POSITION).reshape(-1)
    k_rope = apply_rope_halved(k.reshape(N_KV_HEADS, HEAD_DIM), cos, sin, POSITION).reshape(-1)
    dump("06_q_rope", q_rope, out_dir)
    dump("07_k_rope", k_rope, out_dir)

    # Step 5: attention against single K/V pair (seq_len=1)
    # For seq=1 and no GQA: out[h, :] = v[h, :]  (softmax over single element = 1.0)
    attn_out = v.copy()
    dump("08_attn_out", attn_out, out_dir)

    # Step 6: o projection
    o_out = o_w @ attn_out
    dump("09_o_out", o_out, out_dir)

    # Step 7: post-attn residual + norm (fused_add_rms_norm semantics)
    #   residual = fc_out + o_out
    #   norm_out = rms_norm(residual, post_norm)
    residual_1 = fc_out + o_out
    norm_out = rms_norm(residual_1, post_norm, RMS_NORM_EPS)
    dump("10_residual_1", residual_1, out_dir)
    dump("11_norm_out", norm_out, out_dir)

    # Step 8: MLP
    gate = gate_w @ norm_out
    up   = up_w   @ norm_out
    silu_mul = silu(gate) * up
    down = down_w @ silu_mul
    dump("12_gate", gate, out_dir)
    dump("13_up", up, out_dir)
    dump("14_silu_mul", silu_mul, out_dir)
    dump("15_down", down, out_dir)

    # Step 9: post-MLP residual → final
    final = residual_1 + down
    dump("16_final", final, out_dir)

    print("\n[ref] wrote reference intermediates to /tmp/ref_*.npy")


if __name__ == "__main__":
    main()
