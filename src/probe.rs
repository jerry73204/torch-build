use crate::{
    env::{
        CUDA_HOME, CUDNN_HOME, LIBTORCH, LIBTORCH_BYPASS_VERSION_CHECK, LIBTORCH_CXX11_ABI,
        LIBTORCH_USE_PYTORCH, OUT_DIR, ROCM_HOME, TORCH_VERSION,
    },
    library::{Api, CudaApi, CudaSplitApi, HipApi, Library},
};
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use cfg_if::cfg_if;
use itertools::chain;
use log::warn;
use once_cell::sync::OnceCell;
use std::{
    env,
    io::BufRead,
    path::{Path, PathBuf},
    process::Command,
    str,
};

enum Probe {
    Manual(PathBuf),
    System(PathBuf),
    PyTorch(ProbePyTorch),
    #[allow(unused)]
    Download(PathBuf),
}

struct ProbePyTorch {
    pub include_dirs: Vec<PathBuf>,
    pub lib_dir: PathBuf,
    pub use_cxx11_abi: bool,
}

pub(crate) struct ProbePython {
    pub includes: Vec<PathBuf>,
    pub link_searches: Vec<PathBuf>,
    pub libraries: Vec<String>,
}

/// Probe the installation directory of libtorch and its capabilities.
pub fn probe_libtorch() -> Result<&'static Library> {
    static PROBE: OnceCell<Library> = OnceCell::new();

    PROBE.get_or_try_init(probe_libtorch_private)
}

/// Probe the installation directory of libtorch and its capabilities.
fn probe_libtorch_private() -> Result<Library> {
    let probe = find_or_download_libtorch_dir()?;

    let library = match probe {
        Probe::Manual(libtorch_dir)
        | Probe::System(libtorch_dir)
        | Probe::Download(libtorch_dir) => {
            let lib_dir = libtorch_dir.join("lib");
            let use_cxx11_abi = probe_cxx11_abi();
            let api = probe_cuda_api(&lib_dir);
            let include_dirs: Vec<_> = {
                let base = libtorch_dir.join("include");
                let base_dirs = [
                    base.clone(),
                    base.join("torch").join("csrc").join("api").join("include"),
                    base.join("TH"),
                    base.join("THC"),
                ];
                let thh_include_dir = api.is_hip().then(|| base.join("thh"));
                chain!(base_dirs, thh_include_dir).collect()
            };

            Library {
                api,
                use_cxx11_abi,
                include_dirs,
                lib_dir,
            }
        }
        Probe::PyTorch(library) => {
            let ProbePyTorch {
                include_dirs,
                lib_dir,
                use_cxx11_abi,
            } = library;
            let api = probe_cuda_api(&lib_dir);

            Library {
                include_dirs,
                lib_dir,
                api,
                use_cxx11_abi,
            }
        }
    };

    Ok(library)
}

/// Locate the libtorch directory, or try to download libtorch if it does not exist.
///
/// This function finds the directory in the following order. It
/// returns an error if none of them succeeds.
///
/// 1. Find the directory from `LIBTORCH` environment variable.
/// 2. The host system is Linux and `/usr/lib/libtorch.so` exists.
/// 3. `LIBTORCH_USE_PYTORCH` environment variable is set and the PyTorch is found.
/// 4. If `download-libtorch` feature is set, download from the URL generated by
///   [libtorch_url()](crate::download::libtorch_url) and returns the extracted directory.
///
/// The function is idempotent. It only run once even when the
/// function is called multiple times.
fn find_or_download_libtorch_dir() -> Result<Probe> {
    // Check if LIBTORCH var is set.
    if let Some(dir) = &*LIBTORCH {
        return Ok(Probe::Manual(dir.to_path_buf()));
    }

    // Check if libtorch.so exists on the system.
    #[cfg(target_os = "linux")]
    if Path::new("/usr/lib/libtorch.so").exists() {
        let dir = PathBuf::from("/usr");
        return Ok(Probe::System(dir));
    }

    if *LIBTORCH_USE_PYTORCH {
        let library = probe_pytorch()?;
        return Ok(Probe::PyTorch(library));
    }

    // Try to download the pytorch package
    #[cfg(feature = "download-libtorch")]
    {
        let dir =
            crate::download::download_libtorch().with_context(|| "unable to download libtorch")?;
        return Ok(Probe::Download(dir));
    }

    #[allow(unreachable_code)]
    {
        bail!("unable to find libtorch")
    }
}

/// Return true of host system uses C++11 ABI. It is used to set the
/// `_GLIBCXX_USE_CXX11_ABI` macro.
pub(crate) fn probe_cxx11_abi() -> bool {
    if let Some(val) = *LIBTORCH_CXX11_ABI {
        return val;
    }

    cfg_if! {
        if #[cfg(target_os = "macos")] {
            true
        } else if #[cfg(target_os = "linux")] {
            Path::new(OUT_DIR)
                .join("use_cxx11_abi")
                .exists()
        } else if #[cfg(target_os = "window")] {
            // TODO: check _MSVC_LANG
            true
        } else {
            true
        }
    }
}

fn find_python_interpreter() -> Result<&'static Path> {
    let path = {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    Path::new("python")
                } else {
                    Path::new("python3")
                }
            } else if #[cfg(target_os = "macos")] {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    Path::new("python")
                } else {
                    Path::new("python3")
                }
            } else if #[cfg(target_os = "windows")] {
                Path::from("python.exe")
            } else {
                bail!("Unsupported OS");
            }
        }
    };
    Ok(path)
}

pub(crate) fn probe_python() -> Result<ProbePython> {
    let output = Command::new("python3-config")
        .arg("--includes")
        .arg("--ldflags")
        .arg("--embed")
        .output()?;
    ensure!(output.status.success(), "unable to run `python3-config`");

    let stdout = str::from_utf8(&output.stdout)
        .with_context(|| "unable to parse output of `python3-config`")?;

    let mut includes = vec![];
    let mut link_searches = vec![];
    let mut libraries = vec![];

    for flag in stdout.split([' ', '\n']) {
        let (Some(key), Some(value)) = (flag.get(0..2), flag.get(2..)) else {
            continue;
        };

        match key {
            "-I" => {
                includes.push(PathBuf::from(value));
            }
            "-L" => {
                link_searches.push(PathBuf::from(value));
            }
            "-l" => {
                libraries.push(value.to_string());
            }
            _ => {}
        }
    }

    Ok(ProbePython {
        includes,
        link_searches,
        libraries,
    })
}

fn probe_pytorch() -> Result<ProbePyTorch> {
    const PYTHON_PROBE_PYTORCH_CODE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/pysrc/probe_pytorch.py"
    ));

    let python_interpreter = find_python_interpreter()?;
    let output = Command::new(python_interpreter)
        .arg("-c")
        .arg(PYTHON_PROBE_PYTORCH_CODE)
        .output()
        .with_context(|| format!("error running {python_interpreter:?}"))?;

    let mut use_cxx11_abi = None;
    let mut include_dirs = vec![];
    let mut lib_dir = None;

    for line in output.stdout.lines() {
        let line = line?;

        if let Some(version) = line.strip_prefix("LIBTORCH_VERSION: ") {
            check_pytorch_version(version)?
        } else if let Some(value) = line.strip_prefix("LIBTORCH_CXX11: ") {
            use_cxx11_abi = Some(match value {
                "True" => true,
                "False" => false,
                _ => {
                    bail!("error parsing this line '{line}'");
                }
            });
        } else if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
            include_dirs.push(PathBuf::from(path));
        } else if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
            lib_dir = Some(PathBuf::from(path));
        }
    }

    let use_cxx11_abi =
        use_cxx11_abi.ok_or_else(|| anyhow!("no LIBTORCH_CXX11 returned by python {output:?}"))?;
    let lib_dir =
        lib_dir.ok_or_else(|| anyhow!("no LIBTORCH_LIB returned by python {output:?}"))?;

    Ok(ProbePyTorch {
        include_dirs,
        lib_dir,
        use_cxx11_abi,
    })
}

fn probe_cuda_api(lib_dir: &Path) -> Api {
    let probe_library_file = |name: &str| -> bool {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                lib_dir.join(format!("lib{}.so", name)).exists()
            } else if #[cfg(target_os = "windows")] {
                lib_dir.join(format!("{}.dll", name)).exists()
            } else {
                false
            }
        }
    };

    if let (Some(rocm_home), true) = (&*ROCM_HOME, probe_library_file("torch_hip")) {
        static MIOPEN_HOME: OnceCell<PathBuf> = OnceCell::new();
        let miopen_home = MIOPEN_HOME.get_or_init(|| rocm_home.join("miopen"));

        HipApi {
            rocm_home,
            miopen_home,
        }
        .into()
    } else if let Some(cuda_home) = &*CUDA_HOME {
        if probe_library_file("torch_cuda_cu") && probe_library_file("torch_cuda_cpp") {
            CudaSplitApi {
                cuda_home,
                cudnn_home: CUDNN_HOME.as_deref(),
            }
            .into()
        } else if probe_library_file("torch_cuda") {
            CudaApi {
                cuda_home,
                cudnn_home: CUDNN_HOME.as_deref(),
            }
            .into()
        } else {
            warn!(
                r#"CUDA_HOME is set to "{}", but no CUDA runtime found for libtorch"#,
                cuda_home.display()
            );
            Api::None
        }
    } else {
        Api::None
    }
}

fn check_pytorch_version(version: &str) -> Result<()> {
    if *LIBTORCH_BYPASS_VERSION_CHECK {
        return Ok(());
    }

    let version = version.trim();
    // Typical version number is 2.0.0+cpu or 2.0.0+cu117
    let version = match version.split_once('+') {
        None => version,
        Some((version, _)) => version,
    };

    let torch_version = *TORCH_VERSION;
    if version != torch_version {
        bail!(
            "this tch version expects PyTorch {torch_version}, got {version}, \
               this check can be bypassed by setting the \
               LIBTORCH_BYPASS_VERSION_CHECK environment variable"
        )
    }
    Ok(())
}
