use cfg_if::cfg_if;
use once_cell::sync::{Lazy, OnceCell};
use std::{
    env,
    ffi::OsString,
    path::{PathBuf},
};

pub static TORCH_VERSION: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/TORCH_VERSION"));

pub static LIBTORCH_CXX11_ABI: Lazy<bool> = Lazy::new(|| {
    rerun_env("LIBTORCH_CXX11_ABI")
        .and_then(|val| {
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
        .unwrap_or(true)
});



pub static LIBTORCH: Lazy<Option<PathBuf>> = Lazy::new(|| rerun_env_pathbuf("LIBTORCH"));

pub static TORCH_CUDA_VERSION: Lazy<String> = Lazy::new(|| {
    cfg_if! {
        if #[cfg(any(target_os = "linux", target_os = "window"))] {
            rerun_env_string("TORCH_CUDA_VERSION")
                .map(|val| {
                    val.trim()
                        .to_lowercase()
                        .trim_start_matches("cu")
                        .split('.')
                        .take(2)
                        .fold("cu".to_owned(), |mut acc, curr| {
                            acc += curr;
                            acc
                        })
                })
                .unwrap_or_else(|| "cpu".to_owned())
        } else {
            if let Some(val) = rerun_env("TORCH_CUDA_VERSION") {
                panic!(
                    "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                     binaries with CUDA are only available for Linux and Windows, not: {}.",
                    val
                );
            }
            "cpu".to_owned()
        }
    }
});

pub static LIBTORCH_URL: Lazy<String> = Lazy::new(|| {
    cfg_if! {
        if #[cfg(target_os = "linux")] { format!(
            "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
            *TORCH_CUDA_VERSION, TORCH_VERSION, match TORCH_CUDA_VERSION.as_ref() {
                "cpu" => "%2Bcpu",
                "cu92" => "%2Bcu92",
                "cu101" => "%2Bcu101",
                "cu111" => "%2Bcu111",
                _ => ""
            }
        ) }
        else if #[cfg(target_os = "macos")] { format!(
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
            TORCH_VERSION
        ) }
        else if #[cfg(target_os = "windows")] { format!(
            "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}{}.zip",
            *TORCH_CUDA_VERSION, TORCH_VERSION, match TORCH_CUDA_VERSION.as_ref() {
                "cpu" => "%2Bcpu",
                "cu92" => "%2Bcu92",
                "cu101" => "%2Bcu101",
                "cu111" => "%2Bcu111",
                _ => ""
            }) }
        else { panic!("Unsupported OS") }
    }
});

pub(crate) static LIBTORCH_DIR: OnceCell<PathBuf> = OnceCell::new();

fn rerun_env(name: &str) -> Option<OsString> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var_os(name)
}

fn rerun_env_pathbuf(name: &str) -> Option<PathBuf> {
    Some(rerun_env(name)?.into())
}

fn rerun_env_string(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name).ok()
}
