mod download;
mod globals;

use anyhow::{Context, Result};
use cfg_if::cfg_if;
use once_cell::sync::OnceCell;
use std::{
    env,
    path::{Path, PathBuf},
};

pub use download::*;
pub use globals::*;

pub fn libtorch_dir() -> Result<&'static Path> {
    LIBTORCH_DIR.get_or_try_init(|| {
        let guess = LIBTORCH.to_owned();

        #[cfg(target_os = "linux")]
        let guess = guess.or_else(|| {
            Path::new("/usr/lib/libtorch.so")
                .exists()
                .then(|| PathBuf::from("/usr"))
        });

        cfg_if! {
            if #[cfg(feature = "download-libtorch")] {
                match guess {
                    Some(dir) => Ok(dir),
                    None => {
                        download::download_libtorch().with_context(|| "unable to download libtorch")
                    }
                }
            } else {
                guess.ok_or_else(|| anyhow!("unable to find libtorch"))?;
            }
        }
    })
        .map(|path| path.as_ref())
}

pub fn build() -> Result<()> {
    static BUILD: OnceCell<()> = OnceCell::new();

    BUILD.get_or_try_init(|| -> Result<_> {
        let libtorch_dir = libtorch_dir()?;

        // link libtorch libraries
        let lib_dir = libtorch_dir.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        let probe_file = |name: &str| -> bool {
            cfg_if! {
                if #[cfg(target_os = "linux")] {
                    lib_dir.join(format!("lib{}.so", name)).exists()
                }
                else if #[cfg(target_os = "windows")] {
                    lib_dir.join(format!("{}.dll", name)).exists()
                }
                else { false }
            }
        };

        let use_cuda = probe_file("torch_cuda");
        let use_cuda_cu = probe_file("torch_cuda_cu");
        let use_cuda_cpp = probe_file("torch_cuda_cpp");
        let use_hip = probe_file("torch_hip");

        println!("cargo:rustc-link-lib=static=tch");
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=c10");

        make(&libtorch_dir, use_cuda, use_hip);

        if use_cuda {
            println!("cargo:rustc-link-lib=torch_cuda");
        }
        if use_cuda_cu {
            println!("cargo:rustc-link-lib=torch_cuda_cu");
        }
        if use_cuda_cpp {
            println!("cargo:rustc-link-lib=torch_cuda_cpp");
        }
        if use_hip {
            println!("cargo:rustc-link-lib=torch_hip");
        }
        if use_hip {
            println!("cargo:rustc-link-lib=c10_hip");
        }

        let target = env::var("TARGET").unwrap();
        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }

        Ok(())
    })?;

    Ok(())
}

fn make(libtorch_dir: impl AsRef<Path>, use_cuda: bool, use_hip: bool) {
    let cuda_dependency = if use_cuda || use_hip {
        "libtch/dummy_cuda_dependency.cpp"
    } else {
        "libtch/fake_cuda_dependency.cpp"
    };
    println!("cargo:rerun-if-changed=libtch/torch_api.cpp");
    println!("cargo:rerun-if-changed=libtch/torch_api.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.cpp.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_write.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_resize.h");
    println!("cargo:rerun-if-changed=libtch/stb_image.h");

    cfg_if! {
        if #[cfg(any(target_os = "linux", target_os = "macos"))] {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch_dir.as_ref().join("include"))
                .include(libtorch_dir.as_ref().join("include/torch/csrc/api/include"))
                .flag(&format!(
                    "-Wl,-rpath={}",
                    libtorch_dir.as_ref().join("lib").display()
                ))
                .flag("-std=c++14")
                .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", if *LIBTORCH_CXX11_ABI { "1" } else { "0" }))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        else if #[cfg(target_os = "windows")] {
            // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
            //       Not yet supported by cc=rs.
            //       https://github.com/alexcrichton/cc-rs/issues/323
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch_dir.as_ref().join("include"))
                .include(libtorch_dir.as_ref().join("include/torch/csrc/api/include"))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        else {
            panic!("Unsupported OS")
        }
    }
}
