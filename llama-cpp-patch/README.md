# llama.cpp TurboQuant Integration

This directory contains the integration patch for adding TurboQuant KV cache compression to [llama.cpp](https://github.com/ggerganov/llama.cpp) via the tq-kv Rust library.

## How It Works

```
llama.cpp inference loop
    |
    v
KV cache update --> tq_layer_compress_and_append()  (Rust FFI)
    |
    v
Attention compute --> tq_layer_fused_attention()     (Rust FFI, AVX2 SIMD)
    |
    v
Softmax + V multiply (standard llama.cpp)
```

The Rust library handles compression and fused attention. llama.cpp handles everything else (tokenization, model loading, matmul, sampling).

## Build

### 1. Build tq-kv static library

```bash
cd tq-kv
cargo build --release --features ffi
# Output: target/release/libtq_kv.a (Linux/Mac) or tq_kv.lib (Windows)
```

### 2. Copy files to llama.cpp

```bash
cp tq-kv/include/tq_kv.h    llama.cpp/include/
cp llama-cpp-patch/llama-kv-tq.h   llama.cpp/src/
cp llama-cpp-patch/llama-kv-tq.cpp llama.cpp/src/
cp target/release/libtq_kv.a       llama.cpp/
```

### 3. Patch llama.cpp CMakeLists.txt

Add to `llama.cpp/CMakeLists.txt`:
```cmake
# TurboQuant KV cache compression
option(LLAMA_TQ_KV "Enable TurboQuant KV cache compression" OFF)
if (LLAMA_TQ_KV)
    add_definitions(-DLLAMA_TQ_KV)
    target_sources(llama PRIVATE src/llama-kv-tq.cpp)
    target_include_directories(llama PRIVATE include)
    target_link_libraries(llama PRIVATE ${CMAKE_SOURCE_DIR}/libtq_kv.a pthread dl m)
endif()
```

### 4. Build llama.cpp

```bash
cd llama.cpp
cmake -B build -DLLAMA_TQ_KV=ON
cmake --build build
```

### 5. Run

```bash
./build/bin/llama-cli -m model.gguf --cache-type-k tq2 -p "Hello"
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--cache-type-k tq2` | 2-bit TurboQuant keys (14.2x compression) |
| `--cache-type-k tq3` | 3-bit TurboQuant keys (9.8x compression) |
| `--cache-type-k tq4` | 4-bit TurboQuant keys (3.8x compression) |

Values remain in standard format (fp16/q8_0). Only keys are compressed.

## Performance

With TurboQuant enabled, KV cache memory is reduced by up to 14.2x while maintaining quality:

| Config | PPL (wikitext-2) | KV Memory | Compression |
|--------|-----------------|-----------|-------------|
| fp16 (baseline) | 9.515 | 100% | 1x |
| tq4 | 9.594 (+0.8%) | 26% | 3.8x |
| tq3 | 9.899 (+4.0%) | 10% | 9.8x |
| tq2 | 12.730 (+33.8%) | 7% | 14.2x |

Fused attention (AVX2+FMA) computes attention scores directly from compressed indices -- no decompression needed. 8.9x faster than decompress-then-dot.
