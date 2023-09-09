//! Environment variables and constatns.

use crate::cuda::CudaArch;
use cfg_if::cfg_if;
use once_cell::sync::Lazy;
use std::{
    collections::HashSet,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::Command,
};

/// The list of CUDA architectures given by `TORCH_CUDA_ARCH_LIST` environment variable.
///
/// If `TORCH_CUDA_ARCH_LIST` is not set, the default supported architectures are given.
pub(crate) static TORCH_CUDA_ARCH_LIST: Lazy<HashSet<CudaArch>> = Lazy::new(|| {
    if let Some(val) = rerun_env_string("TORCH_CUDA_ARCH_LIST") {
        CudaArch::parse_list(&val)
            .unwrap_or_else(|_| {
                panic!(
                    r#"unable to parse environment variable TORCH_CUDA_ARCH_LIST = "{}""#,
                    val
                )
            })
            .into_iter()
            .collect()
    } else {
        let text = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/TORCH_CUDA_ARCH_LIST"));
        CudaArch::parse_list(text)
            .expect("unable to load TORCH_CUDA_ARCH_LIST file")
            .into_iter()
            .collect()
    }
});

pub(crate) static OUT_DIR: &str = env!("OUT_DIR");

pub(crate) static TARGET: Lazy<Option<String>> = Lazy::new(|| rerun_env_string("TARGET"));

/// The supported libtorch version.
pub static TORCH_VERSION: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/TORCH_VERSION"));

/// The value of `LIBTORCH_CXX11_ABI` environment variable.
pub static LIBTORCH_CXX11_ABI: Lazy<Option<bool>> = Lazy::new(|| {
    rerun_env("LIBTORCH_CXX11_ABI").and_then(|val| {
        cfg_if! {
            if #[cfg(unix)] {
                use std::os::unix::ffi::OsStrExt;
                match val.as_bytes() {
                    b"1" => Some(true),
                    b"0" => Some(false),
                    _ => {
                        // warn
                        None
                    }
                }
            }
            else {
                match val.to_str() {
                    Some("1") => Some(true),
                    Some("0") => Some(false),
                    _ => None,
                }
            }
        }
    })
});

/// The value of `LIBTORCH` environment variable.
pub static LIBTORCH: Lazy<Option<PathBuf>> = Lazy::new(|| rerun_env_pathbuf("LIBTORCH"));

/// The value of `TORCH_CUDA_VERSION` environment variable.
pub static TORCH_CUDA_VERSION: Lazy<Option<String>> =
    Lazy::new(|| rerun_env_string("TORCH_CUDA_VERSION"));

/// The value of `CUDNN_HOME` environment variable, or `CUDNN_PATH` if `CUDNN_HOME` is not set.
pub static CUDNN_HOME: Lazy<Option<PathBuf>> =
    Lazy::new(|| rerun_env_pathbuf("CUDNN_HOME").or_else(|| rerun_env_pathbuf("CUDNN_PATH")));

/// The value of `ROCM_HOME` environment variable, or `ROCM_PATH` if `ROCM_HOME` is not set.
pub static ROCM_HOME: Lazy<Option<PathBuf>> = Lazy::new(|| {
    let guess = rerun_env_pathbuf("ROCM_HOME")
        .or_else(|| rerun_env_pathbuf("ROCM_PATH"))
        .map(PathBuf::from);

    #[cfg(unix)]
    let guess = guess.or_else(|| {
        Command::new("sh")
            .arg("-c")
            .arg("which hipcc | xargs readlink -f")
            .output()
            .ok()
            .and_then(|output| output.status.success().then_some(output.stdout))
            .and_then(|stdout| {
                use std::os::unix::ffi::OsStrExt;

                // strip trailing line breaks
                let stdout = stdout.strip_suffix(b"\n")?;
                let path = Path::new(OsStr::from_bytes(stdout));

                let dir = path.parent()?.parent()?;
                (dir.file_name()? == "hip")
                    .then(|| dir.parent())
                    .flatten()
                    .map(PathBuf::from)
            })
    });

    #[cfg(unix)]
    let guess = guess.or_else(|| {
        let dir = PathBuf::from("/opt/rocm");
        dir.exists().then_some(dir)
    });

    guess
});

/// The CUDA installation directory on host system.
///
/// The value is determined in the following order.
///
/// 1. `CUDA_HOME` environment variable.
/// 2. `CUDA_PATH` environment variable.
/// 3. The path returned by `nvcc` if on Liunx or Mac.
/// 4. `/usr/local/cuda` if on Debian or Ubuntu and the directory exists.
pub static CUDA_HOME: Lazy<Option<PathBuf>> = Lazy::new(|| {
    use os_info::Type::*;

    let guess = rerun_env_pathbuf("CUDA_HOME")
        .or_else(|| rerun_env_pathbuf("CUDA_PATH"))
        .map(PathBuf::from);

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let guess = guess.or_else(|| {
        Command::new("which")
            .arg("nvcc")
            .output()
            .ok()
            .and_then(|output| output.status.success().then_some(output.stdout))
            .and_then(|stdout| {
                use std::os::unix::ffi::OsStrExt;

                // strip trailing line breaks
                let stdout = stdout.strip_suffix(b"\n")?;
                let path = Path::new(OsStr::from_bytes(stdout));
                let dir = path.parent()?.parent()?.into();
                Some(dir)
            })
    });

    match os_info::get().os_type() {
        Debian | Ubuntu => guess.or_else(|| {
            let dir = PathBuf::from("/usr/local/cuda");
            dir.exists().then_some(dir)
        }),
        _ => guess,
    }
});

fn rerun_env(name: &str) -> Option<OsString> {
    println!("cargo:rerun-if-env-changed={}", name);
    std::env::var_os(name)
}

fn rerun_env_pathbuf(name: &str) -> Option<PathBuf> {
    Some(rerun_env(name)?.into())
}

fn rerun_env_string(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={}", name);
    std::env::var(name).ok()
}
