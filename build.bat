@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set LIBCLANG_PATH=C:\Program Files\LLVM\bin
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CMAKE_GENERATOR=Ninja
set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor
set CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING=1
set PATH=C:\Program Files\CMake\bin;C:\Users\onurg\AppData\Local\Microsoft\WinGet\Links;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%

if "%1"=="release" (
    cargo build --release
) else if "%1"=="check" (
    cargo check
) else if "%1"=="test" (
    cargo test %2 %3 %4
) else (
    cargo build
)
