//! Build script — compile CUDA kernels to PTX via nvcc.
//!
//! Multi-arch: compiles kernels/*.cu for sm_75/80/86/89/90 (configurable).
//! At runtime, the best matching PTX is loaded for the detected GPU.
//!
//! Override target arches: TQ_CUDA_ARCHES=86 (comma-separated, default: all)
//! Feature-gated: only runs with `--features cuda`.

#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};

fn main() {
    // Only compile CUDA kernels when the cuda feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    // Always rerun if kernels change
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TQ_CUDA_ARCHES");
}

#[cfg(feature = "cuda")]
fn parse_cuda_arches() -> Vec<u32> {
    match env::var("TQ_CUDA_ARCHES") {
        Ok(val) => {
            let arches: Vec<u32> = val.split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();
            if arches.is_empty() {
                eprintln!("WARNING: TQ_CUDA_ARCHES set but no valid arches parsed, using defaults");
                vec![75, 80, 86, 89, 90]
            } else {
                eprintln!("TQ_CUDA_ARCHES override: {:?}", arches);
                arches
            }
        }
        Err(_) => vec![75, 80, 86, 89, 90],
    }
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = Path::new("kernels");

    // Find nvcc
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(windows) {
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    let nvcc = if cfg!(windows) {
        PathBuf::from(&cuda_path).join("bin").join("nvcc.exe")
    } else {
        PathBuf::from(&cuda_path).join("bin").join("nvcc")
    };

    if !nvcc.exists() {
        eprintln!("WARNING: nvcc not found at {:?}, CUDA kernels will not be compiled", nvcc);
        eprintln!("  Set CUDA_PATH or CUDA_HOME environment variable");
        generate_empty_ptx_module(&out_dir);
        return;
    }

    eprintln!("Found nvcc at {:?}", nvcc);

    // Target architectures
    let arches = parse_cuda_arches();
    eprintln!("Target arches: {:?}", arches);

    // Find MSVC cl.exe path (Windows only, resolved once)
    let ccbin_path = if cfg!(windows) { find_msvc_ccbin() } else { None };

    // Collect .cu files
    let cu_files: Vec<PathBuf> = std::fs::read_dir(kernels_dir)
        .expect("kernels/ directory not found")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "cu" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if cu_files.is_empty() {
        eprintln!("WARNING: No .cu files found in kernels/");
        generate_empty_ptx_module(&out_dir);
        return;
    }

    eprintln!("Found {} .cu files, compiling for {} arch(es) ({} total)",
        cu_files.len(), arches.len(), cu_files.len() * arches.len());

    // Compile each .cu for each arch → (kernel_name, sm, ptx_path)
    let mut ptx_entries: Vec<(String, u32, PathBuf)> = Vec::new();

    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_str().unwrap();

        for &sm in &arches {
            let ptx_path = out_dir.join(format!("{}_sm{}.ptx", stem, sm));
            let arch_flag = format!("-arch=sm_{}", sm);

            let mut cmd = std::process::Command::new(&nvcc);
            cmd.args(&[
                "--ptx",
                &arch_flag,
                "-O3",
                "--use_fast_math",
                "-I", kernels_dir.to_str().unwrap(),
                "-o", ptx_path.to_str().unwrap(),
                cu_file.to_str().unwrap(),
            ]);

            if let Some(ref ccbin) = ccbin_path {
                cmd.arg("-ccbin").arg(ccbin);
            }

            let status = cmd.status();

            match status {
                Ok(s) if s.success() => {
                    ptx_entries.push((stem.to_string(), sm, ptx_path));
                }
                Ok(s) => {
                    eprintln!("WARNING: nvcc failed for {}_sm{} (exit code {:?})", stem, sm, s.code());
                }
                Err(e) => {
                    eprintln!("WARNING: Failed to run nvcc for {}_sm{}: {}", stem, sm, e);
                }
            }
        }
    }

    // Generate ptx_generated.rs with multi-arch PTX selection
    generate_ptx_module_multi_arch(&out_dir, &ptx_entries, &arches);
}

/// Find MSVC cl.exe for nvcc host compilation on Windows.
#[cfg(feature = "cuda")]
fn find_msvc_ccbin() -> Option<String> {
    let msvc_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64",
    ];
    for p in &msvc_paths {
        let cl = Path::new(p).join("cl.exe");
        if cl.exists() {
            return Some(p.to_string());
        }
    }
    None
}

/// Generate ptx_generated.rs with multi-arch PTX constants and selection helpers.
#[cfg(feature = "cuda")]
fn generate_ptx_module_multi_arch(
    out_dir: &Path,
    entries: &[(String, u32, PathBuf)],
    _arches: &[u32],
) {
    use std::collections::BTreeMap;

    let mut code = String::from(
        "// Auto-generated PTX module — do not edit manually.\n\
         // Generated by build.rs from kernels/*.cu\n\
         // Multi-arch: per-GPU PTX selected at runtime via best_compiled_arch()\n\n"
    );

    // 1. Generate per-arch PTX constants: PTX_QMATMUL_SM86, PTX_ELEMENTWISE_SM75, etc.
    for (name, sm, ptx_path) in entries {
        let const_name = format!("PTX_{}_SM{}", name.to_uppercase(), sm);
        code.push_str(&format!(
            "pub const {}: &str = include_str!(\"{}\");\n",
            const_name,
            ptx_path.to_str().unwrap().replace('\\', "/"),
        ));
    }
    code.push('\n');

    // 2. Determine which arches actually compiled successfully
    let mut arch_groups: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (name, sm, _) in entries {
        arch_groups.entry(*sm).or_default().push(name.clone());
    }

    let compiled_arches: Vec<u32> = arch_groups.keys().copied().collect();
    code.push_str(&format!(
        "/// GPU architectures with compiled PTX available.\n\
         pub const COMPILED_ARCHES: &[u32] = &[{}];\n\n",
        compiled_arches.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
    ));

    // 3. Generate static module arrays per arch
    for (&sm, names) in &arch_groups {
        code.push_str(&format!(
            "static ARCH_{}_MODULES: &[(&str, &str)] = &[\n",
            sm
        ));
        for name in names {
            code.push_str(&format!(
                "    (\"{}\", PTX_{}_SM{}),\n",
                name,
                name.to_uppercase(),
                sm
            ));
        }
        code.push_str("];\n\n");
    }

    // 4. Generate ptx_sources_for_arch() selector
    code.push_str(
        "/// Get PTX sources for a specific compiled architecture.\n\
         /// Returns (module_name, ptx_source) pairs.\n\
         pub fn ptx_sources_for_arch(arch: u32) -> &'static [(&'static str, &'static str)] {\n    \
             match arch {\n"
    );
    for &sm in arch_groups.keys() {
        code.push_str(&format!("        {} => ARCH_{}_MODULES,\n", sm, sm));
    }
    code.push_str("        _ => &[],\n    }\n}\n\n");

    // 5. Generate best_compiled_arch() helper
    code.push_str(
        "/// Find the best compiled arch for a given GPU compute capability.\n\
         /// Returns the highest compiled arch <= the GPU's sm version.\n\
         /// Falls back to lowest available arch if GPU is older than all compiled arches.\n\
         pub fn best_compiled_arch(sm_major: u32, sm_minor: u32) -> u32 {\n    \
             let sm = sm_major * 10 + sm_minor;\n    \
             let mut best = COMPILED_ARCHES[0]; // fallback to lowest\n    \
             for &arch in COMPILED_ARCHES {\n        \
                 if arch <= sm { best = arch; }\n    \
             }\n    \
             best\n\
         }\n"
    );

    // Write
    let ptx_rs = out_dir.join("ptx_generated.rs");
    std::fs::write(&ptx_rs, &code).expect("Failed to write ptx_generated.rs");
    eprintln!("Generated {:?}: {} PTX constants across {} arch(es)",
        ptx_rs, entries.len(), arch_groups.len());
}

#[allow(dead_code)]
fn generate_empty_ptx_module(out_dir: &std::path::Path) {
    let code = "\
// No CUDA kernels compiled (nvcc not found or cuda feature disabled).\n\
pub const COMPILED_ARCHES: &[u32] = &[];\n\
pub fn ptx_sources_for_arch(_arch: u32) -> &'static [(&'static str, &'static str)] { &[] }\n\
pub fn best_compiled_arch(_sm_major: u32, _sm_minor: u32) -> u32 { 0 }\n";
    let ptx_rs = out_dir.join("ptx_generated.rs");
    std::fs::write(&ptx_rs, code).expect("Failed to write empty ptx_generated.rs");
}
