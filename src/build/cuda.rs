use anyhow::{ensure, Context as _, Result};
use cfg_if::cfg_if;
use log::warn;
use std::{
    path::{Path, PathBuf},
    process::Command,
    str,
};

#[derive(Debug, Clone)]
pub struct CudaExtension {
    link_python: bool,
    includes: Vec<PathBuf>,
    link_searches: Vec<PathBuf>,
    libraries: Vec<String>,
    sources: Vec<PathBuf>,
}

impl CudaExtension {
    pub fn new() -> Self {
        Self {
            link_python: false,
            includes: vec![],
            sources: vec![],
            link_searches: vec![],
            libraries: vec![],
        }
    }

    pub fn link_python(&mut self, enabled: bool) -> &mut Self {
        self.link_python = enabled;
        self
    }

    pub fn include<P>(&mut self, path: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.includes.push(path.as_ref().to_owned());
        self
    }

    pub fn includes<P>(&mut self, paths: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self.includes
            .extend(paths.into_iter().map(|p| p.as_ref().to_owned()));
        self
    }

    pub fn source<P>(&mut self, path: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.sources.push(path.as_ref().to_owned());
        self
    }

    pub fn sources<P>(&mut self, paths: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self.sources
            .extend(paths.into_iter().map(|p| p.as_ref().to_owned()));
        self
    }

    pub fn link_search<P>(&mut self, path: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.link_searches.push(path.as_ref().to_owned());
        self
    }

    pub fn link_searches<P>(&mut self, paths: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self.link_searches
            .extend(paths.into_iter().map(|p| p.as_ref().to_owned()));
        self
    }

    pub fn library<P>(&mut self, name: P) -> &mut Self
    where
        P: AsRef<str>,
    {
        self.libraries.push(name.as_ref().to_owned());
        self
    }

    pub fn libraries<P>(&mut self, names: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: AsRef<str>,
    {
        self.libraries
            .extend(names.into_iter().map(|p| p.as_ref().to_owned()));
        self
    }

    /// Compile and link CUDA source code. This is a shorthand for
    /// [configure()](CudaExtension::configure) and then
    /// [link()](CudaExtension::link).
    pub fn build(&self, name: &str) -> Result<()> {
        let mut build = cc::Build::new();
        self.configure(&mut build)?;
        build.try_compile(name)?;
        self.link()?;
        Ok(())
    }

    /// Configure the [cc::Build] to compile CUDA source code.
    pub fn configure(&self, build: &mut cc::Build) -> Result<()> {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                self.configure_unix(build)?;
            } else if #[cfg(target_os = "windows")] {
                bail!("Unsupported OS")l
            } else {
                bail!("Unsupported OS")l
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn configure_unix(&self, build: &mut cc::Build) -> Result<()> {
        let Self {
            link_python: use_python,
            ref includes,
            ref link_searches,
            ref libraries,
            ref sources,
        } = *self;

        let libtorch = crate::probe::probe_libtorch()?;
        ensure!(
            libtorch.is_cuda_api_available(),
            "CUDA runtime is not supported by PyTorch"
        );
        const USE_CUDA_API: bool = true;

        let cxx11_abi_flag = if libtorch.use_cxx11_abi { "1" } else { "0" };
        let cuda_arches = crate::cuda::cuda_arches()?;

        build
            .cuda(true)
            .pic(true)
            .includes(libtorch.include_paths(USE_CUDA_API)?)
            .includes(includes)
            .flag("-std=c++14")
            .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi_flag))
            .files(sources);

        // specify CUDA architecture flags
        cuda_arches.iter().for_each(|arch| {
            build.flag(&arch.nvcc_flag());
        });

        // utilities
        let add_link_search = |build: &mut cc::Build, path: &Path| {
            build
                .flag("-Xlinker")
                .flag(&format!("-Wl,-rpath={}", path.display()));
        };
        let add_library = |build: &mut cc::Build, name: &str| {
            build.flag(&format!("-l{name}"));
        };

        // link libtorch
        libtorch.link_paths(true)?.for_each(|path| {
            add_link_search(build, &path);
        });
        libtorch.libraries(true, use_python)?.for_each(|library| {
            add_library(build, library);
        });

        // link user-specified libraries
        libraries.iter().for_each(|library| {
            add_library(build, library);
        });
        link_searches.iter().for_each(|path| {
            add_link_search(build, path);
        });

        // link python
        if use_python {
            configure_python_libs_unix(build)?;
        }

        Ok(())
    }

    pub fn link(&self) -> Result<()> {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                self.link_unix()?
            } else if #[cfg(target_os = "windows")] {
                bail!("Unsupported OS")
            } else {
                bail!("Unsupported OS")
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn link_unix(&self) -> Result<()> {
        use crate::build::utils::{print_cargo_link_library, print_cargo_link_search};

        let Self {
            link_python,
            ref link_searches,
            ref libraries,
            ..
        } = *self;

        let libtorch = crate::probe::probe_libtorch()?;
        ensure!(
            libtorch.is_cuda_api_available(),
            "CUDA runtime is not supported by PyTorch"
        );

        // link libtorch
        libtorch.link_paths(true)?.for_each(|path| {
            print_cargo_link_search(&path);
        });
        libtorch.libraries(true, link_python)?.for_each(|library| {
            print_cargo_link_library(library);
        });

        // link user-specified libraries
        libraries.iter().for_each(|library| {
            print_cargo_link_library(library);
        });
        link_searches.iter().for_each(|path| {
            print_cargo_link_search(path);
        });

        // link python
        if link_python {
            link_python_libs_unix()?;
        }

        Ok(())
    }
}

impl Default for CudaExtension {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn configure_python_libs_unix(build: &mut cc::Build) -> Result<()> {
    let output = Command::new("python3-config")
        .arg("--includes")
        .arg("--ldflags")
        .arg("--embed")
        .output()?;
    ensure!(output.status.success(), "unable to run `python3-config`");
    let stdout = str::from_utf8(&output.stdout)
        .with_context(|| "unable to parse output `python3-config`")?;
    stdout
        .split(&[' ', '\n'][..])
        .for_each(|flag| match flag.get(0..2) {
            Some("-I") => {
                let path = &flag[2..];
                build.include(path);
            }
            Some("-L") => {
                let path = &flag[2..];
                build.flag("-Xlinker").flag(&format!("-Wl,-rpath={path}"));
            }
            Some("-l") => {
                let library = &flag[2..];
                build.flag(&format!("-l{library}"));
            }
            _ => {
                warn!("ignore `python3-config` flag {}", flag);
            }
        });

    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn link_python_libs_unix() -> Result<()> {
    use crate::build::utils::{print_cargo_link_library, print_cargo_link_search};

    let output = Command::new("python3-config")
        .arg("--includes")
        .arg("--ldflags")
        .arg("--embed")
        .output()?;
    ensure!(output.status.success(), "unable to run `python3-config`");
    let stdout = str::from_utf8(&output.stdout)
        .with_context(|| "unable to parse output `python3-config`")?;
    stdout
        .split(&[' ', '\n'][..])
        .for_each(|flag| match flag.get(0..2) {
            Some("-I") => {
                // no-op
            }
            Some("-L") => {
                let path = &flag[2..];
                print_cargo_link_search(path);
            }
            Some("-l") => {
                let library = &flag[2..];
                print_cargo_link_library(library);
            }
            _ => {
                warn!("ignore `python3-config` flag {}", flag);
            }
        });

    Ok(())
}
