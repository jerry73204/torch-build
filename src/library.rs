//! Libtorch installation and capabilities.

use crate::{env::TARGET, utils::IteratorExt as _};
use anyhow::{bail, Result};
use cfg_if::cfg_if;
use itertools::chain;
use std::{
    iter,
    path::{Path, PathBuf},
    str,
};

/// The information of libtorch installation and its capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Library {
    /// The directories containing header files.
    pub include_dirs: Vec<PathBuf>,

    /// The directory containing library files.
    pub lib_dir: PathBuf,

    /// The CUDA API variants.
    pub api: Api,

    /// True if host system uses C++11 ABI.
    pub use_cxx11_abi: bool,
}

impl Library {
    /// Generate include paths that is passed to C++ compiler.
    ///
    /// The `use_cuda_api` determines whether to enable CUDA API.
    /// - `true`: CUDA is mandatory.
    /// - `false`: CUDA is disabled.
    /// - `None`: CUDA is enabled if supported.
    pub fn include_paths(
        &self,
        use_cuda_api: impl Into<Option<bool>>,
    ) -> Result<impl Iterator<Item = PathBuf>> {
        let Self {
            include_dirs: base_includes,
            api,
            ..
        } = self;
        let use_cuda_api = use_cuda_api
            .into()
            .unwrap_or_else(|| self.is_cuda_api_available());

        let extra_includes = if use_cuda_api {
            match api {
                Api::Hip(HipApi {
                    rocm_home,
                    miopen_home,
                    ..
                }) => {
                    let rocm_include = rocm_home.join("include");
                    let miopen_include = miopen_home.join("include");

                    [rocm_include, miopen_include].into_iter().boxed()
                }
                Api::Cuda(CudaApi {
                    cuda_home,
                    cudnn_home,
                })
                | Api::CudaSplit(CudaSplitApi {
                    cuda_home,
                    cudnn_home,
                }) => {
                    let cuda_include = cuda_home.join("include");
                    let cudnn_include = cudnn_home.map(|path| path.join("include"));
                    chain!([cuda_include], cudnn_include).boxed()
                }
                Api::None => bail!("CUDA runtime is available"),
            }
        } else {
            iter::empty().boxed()
        };

        let all_includes = chain!(base_includes.clone(), extra_includes);

        #[cfg(target_os = "linux")]
        let all_includes = all_includes.filter(|path| path != Path::new("/usr/include"));

        Ok(all_includes)
    }

    /// Generate link paths that is passed to C++ compiler.
    ///
    /// The `use_cuda_api` determines whether to enable CUDA API.
    /// - `true`: CUDA is mandatory.
    /// - `false`: CUDA is disabled.
    /// - `None`: CUDA is enabled if supported.
    pub fn link_paths(
        &self,
        use_cuda_api: impl Into<Option<bool>>,
    ) -> Result<impl Iterator<Item = PathBuf>> {
        let Self {
            lib_dir: libtorch_lib_dir,
            api,
            ..
        } = self;
        let use_cuda_api = use_cuda_api
            .into()
            .unwrap_or_else(|| self.is_cuda_api_available());
        let lib_dir = libtorch_lib_dir;
        let extra_dirs = if use_cuda_api {
            match api {
                Api::Hip(HipApi { rocm_home, .. }) => iter::once(rocm_home.join("lib")).boxed(),
                Api::Cuda(CudaApi {
                    cuda_home,
                    cudnn_home,
                })
                | Api::CudaSplit(CudaSplitApi {
                    cuda_home,
                    cudnn_home,
                }) => {
                    cfg_if! {
                        if #[cfg(target_os = "windows")] {
                            let cuda_lib_dir = cuda_home.un.join("lib").join("x64");
                            iter::once(cuda_lib_dir).boxed()
                        }
                        else if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                            let cuda_lib_dir = {
                                let guess1 = cuda_home.join("lib64");
                                let guess2 = cuda_home.join("lib");
                                match (guess1.exists(), guess2.exists()) {
                                    (true, _) => guess1,
                                    (false, true) => guess2,
                                    (false, false) => bail!("TODO"),
                                }
                            };
                            let cudnn_lib_dir = if let Some(cudnn_home) =  cudnn_home {
                                let guess1 = cudnn_home.join("lib64");
                                let guess2 = cudnn_home.join("lib");
                                let dir = match (guess1.exists(), guess2.exists()) {
                                    (true, _) => guess1,
                                    (false, true) => guess2,
                                    (false, false) => bail!("TODO"),
                                };
                                Some(dir)
                            } else {
                                None
                            };

                            chain!([cuda_lib_dir], cudnn_lib_dir).boxed()
                        }
                        else {
                            bail!("Unsupported OS");
                        }
                    }
                }
                Api::None => bail!("CUDA runtime is available"),
            }
        } else {
            iter::empty().boxed()
        };

        let all_paths = chain!([lib_dir.clone()], extra_dirs);

        Ok(all_paths)
    }

    /// Generate linked libraries that is passed to C++ compiler.
    ///
    /// The `use_cuda_api` determines whether to enable CUDA API.
    /// - `true`: CUDA is mandatory.
    /// - `false`: CUDA is disabled.
    /// - `None`: CUDA is enabled if supported.
    pub fn libraries(
        &self,
        use_cuda_api: impl Into<Option<bool>>,
        use_python: bool,
    ) -> Result<impl Iterator<Item = &'static str>> {
        let Self { api, .. } = self;
        let use_cuda_api = use_cuda_api
            .into()
            .unwrap_or_else(|| self.is_cuda_api_available());
        let base_libraries = ["c10", "torch_cpu", "torch"];
        let python_library = use_python.then_some("torch_python");
        let base_cuda_libraries = ["cudart", "c10_cuda"];

        let cuda_libraries = if use_cuda_api {
            match api {
                Api::None => bail!("CUDA runtime is available"),
                Api::Hip(_) => {
                    [
                        // TODO: check ROCM version
                        "amdhip64", // ROCM version >= 3.5
                        // "hip_hcc", // for ROCM version < 3.5
                        "c10_hip",
                        "torch_hip",
                    ]
                    .into_iter()
                    .boxed()
                }
                Api::Cuda(_) => chain!(base_cuda_libraries, ["torch_cuda"]).boxed(),
                Api::CudaSplit(_) => {
                    chain!(base_cuda_libraries, ["torch_cuda_cu", "torch_cuda_cpp"]).boxed()
                }
            }
        } else {
            iter::empty().boxed()
        };

        let gomp = TARGET.as_ref().and_then(|target| {
            let ok = !target.contains("msvc") && !target.contains("apple");
            ok.then_some("gomp")
        });

        Ok(chain!(base_libraries, python_library, cuda_libraries, gomp))
    }

    /// Check if CUDA runtime is available.
    pub fn is_cuda_api_available(&self) -> bool {
        self.api.is_cuda_api_available()
    }
}

/// CUDA API variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Api {
    None,
    Hip(HipApi),
    Cuda(CudaApi),
    CudaSplit(CudaSplitApi),
}

impl Api {
    /// Check if CUDA runtime is available.
    pub fn is_cuda_api_available(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn cuda_home_dir(&self) -> Option<&Path> {
        Some(match self {
            Self::None | Self::Hip(_) => return None,
            Self::Cuda(api) => api.cuda_home,
            Self::CudaSplit(api) => api.cuda_home,
        })
    }

    pub fn cudnn_home_dir(&self) -> Option<&Path> {
        match self {
            Self::None | Self::Hip(_) => None,
            Self::Cuda(api) => api.cudnn_home,
            Self::CudaSplit(api) => api.cudnn_home,
        }
    }

    pub fn cuda_include_dir(&self) -> Option<PathBuf> {
        Some(self.cuda_home_dir()?.join("include"))
    }

    pub fn cuda_library_dir(&self) -> Option<PathBuf> {
        Some(self.cuda_home_dir()?.join("lib64"))
    }

    /// Returns `true` if the api is [`Hip`].
    ///
    /// [`Hip`]: Api::Hip
    #[must_use]
    pub fn is_hip(&self) -> bool {
        matches!(self, Self::Hip(..))
    }
}

impl From<HipApi> for Api {
    fn from(from: HipApi) -> Self {
        Self::Hip(from)
    }
}

impl From<CudaApi> for Api {
    fn from(from: CudaApi) -> Self {
        Self::Cuda(from)
    }
}

impl From<CudaSplitApi> for Api {
    fn from(from: CudaSplitApi) -> Self {
        Self::CudaSplit(from)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HipApi {
    pub rocm_home: &'static Path,
    pub miopen_home: &'static Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaApi {
    pub cuda_home: &'static Path,
    pub cudnn_home: Option<&'static Path>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaSplitApi {
    pub cuda_home: &'static Path,
    pub cudnn_home: Option<&'static Path>,
}
