use anyhow::{ensure, Context as _, Result};
use cfg_if::cfg_if;
use log::warn;
use std::{
    path::{Path, PathBuf},
    process::Command,
    str,
};

pub use cpp::*;
mod cpp {
    use super::*;

    /// Construct the [cc::Build] to compile C++ source code.
    pub fn build_cpp<B, SourcePath, SourcePathIter>(
        build: &mut cc::Build,
        use_cuda_api: B,
        link_python: bool,
        cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
    ) -> Result<()>
    where
        B: Into<Option<bool>>,
        SourcePath: AsRef<Path>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
    {
        build_cpp_ext::<_, _, PathBuf, PathBuf, String, _, _, _, _>(
            build,
            use_cuda_api,
            link_python,
            cargo_commands,
            sources,
            [],
            [],
            [],
        )
    }

    /// Construct the [cc::Build] to compile C++ source code with additional options.
    pub fn build_cpp_ext<
        B,
        SourcePath,
        IncludePath,
        LinkPath,
        Library,
        SourcePathIter,
        IncludePathIter,
        LinkPathIter,
        LibraryIter,
    >(
        build: &mut cc::Build,
        use_cuda_api: B,
        link_python: bool,
        cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
        include_paths: IncludePathIter,
        link_paths: LinkPathIter,
        libraries: LibraryIter,
    ) -> Result<()>
    where
        B: Into<Option<bool>>,
        SourcePath: AsRef<Path>,
        IncludePath: AsRef<Path>,
        LinkPath: AsRef<Path>,
        Library: AsRef<str>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
        IncludePathIter: IntoIterator<Item = IncludePath>,
        LinkPathIter: IntoIterator<Item = LinkPath>,
        LibraryIter: IntoIterator<Item = Library>,
    {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                build_cpp_ext_unix(
                    build,
                    use_cuda_api,
                    link_python,
                    cargo_commands,
                    sources,
                    include_paths,
                    link_paths,
                    libraries
                )?
            } else if #[cfg(target_os = "windows")] {
                // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
                //       Not yet supported by cc=rs.
                //       https://github.com/alexcrichton/cc-rs/issues/323
                let libtorch = crate::probe::probe_libtorch()?;
                let use_cuda_api = use_cuda_api.into();
                build.cpp(true)
                    .pic(true)
                    .includes(libtorch.include_paths(use_cuda_abi)?)
                    .includes(include_paths)
                    .files(sources);
                build
            } else {
                bail!("Unsupported OS")
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn build_cpp_ext_unix<
        B,
        SourcePath,
        IncludePath,
        LinkPath,
        Library,
        SourcePathIter,
        IncludePathIter,
        LinkPathIter,
        LibraryIter,
    >(
        build: &mut cc::Build,
        use_cuda_api: B,
        use_python: bool,
        mut cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
        include_paths: IncludePathIter,
        link_paths: LinkPathIter,
        libraries: LibraryIter,
    ) -> Result<()>
    where
        B: Into<Option<bool>>,
        SourcePath: AsRef<Path>,
        IncludePath: AsRef<Path>,
        LinkPath: AsRef<Path>,
        Library: AsRef<str>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
        IncludePathIter: IntoIterator<Item = IncludePath>,
        LinkPathIter: IntoIterator<Item = LinkPath>,
        LibraryIter: IntoIterator<Item = Library>,
    {
        let libtorch = crate::probe::probe_libtorch()?;
        let use_cuda_api = use_cuda_api.into();
        let cxx11_abi_flag = if libtorch.use_cxx11_abi { "1" } else { "0" };

        build
            .cpp(true)
            .pic(true)
            .includes(libtorch.include_paths(use_cuda_api)?)
            .includes(include_paths)
            .flag("-std=c++14")
            .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi_flag))
            .files(sources);

        // link libtorch
        libtorch.link_paths(use_cuda_api)?.for_each(|path| {
            add_link_path_unix(build, &path, &mut cargo_commands);
        });
        libtorch
            .libraries(use_cuda_api, use_python)?
            .for_each(|library| {
                add_library_unix(build, library, &mut cargo_commands);
            });

        // link user-specified libraries
        link_paths.into_iter().for_each(|path| {
            add_link_path_unix(build, path.as_ref(), &mut cargo_commands);
        });
        libraries.into_iter().for_each(|lib| {
            add_library_unix(build, lib.as_ref(), &mut cargo_commands);
        });

        // link python
        if use_python {
            link_python_libs_unix(build, &mut cargo_commands)?;
        }

        Ok(())
    }

    // utility functions

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn link_python_libs_unix(
        build: &mut cc::Build,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) -> Result<()> {
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
                    add_link_path_unix(build, Path::new(path), cargo_commands);
                }
                Some("-l") => {
                    let library = &flag[2..];
                    add_library_unix(build, library, cargo_commands);
                }
                _ => {
                    warn!("ignore `python3-config` flag {}", flag);
                }
            });

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn add_link_path_unix(
        build: &mut cc::Build,
        path: &Path,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) {
        build.flag(&format!("-Wl,-rpath={}", path.display()));
        if let Some(cargo_commands) = cargo_commands {
            cargo_commands.push(format!("cargo:rustc-link-search=native={}", path.display()));
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn add_library_unix(
        build: &mut cc::Build,
        library: &str,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) {
        build.flag(&format!("-l{}", library));
        if let Some(cargo_commands) = cargo_commands {
            cargo_commands.push(format!("cargo:rustc-link-lib={}", library));
        }
    }
}

pub use cuda::*;
mod cuda {
    use super::*;

    /// Construct the [cc::Build] to compile CUDA source code.
    pub fn build_cuda<SourcePath, SourcePathIter>(
        build: &mut cc::Build,
        use_python: bool,
        cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
    ) -> Result<()>
    where
        SourcePath: AsRef<Path>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
    {
        build_cuda_ext::<_, PathBuf, PathBuf, String, _, _, _, _>(
            build,
            use_python,
            cargo_commands,
            sources,
            [],
            [],
            [],
        )
    }

    /// Construct the [cc::Build] to compile CUDA source code with additional options.
    pub fn build_cuda_ext<
        SourcePath,
        IncludePath,
        LinkPath,
        Library,
        SourcePathIter,
        IncludePathIter,
        LinkPathIter,
        LibraryIter,
    >(
        build: &mut cc::Build,
        use_python: bool,
        cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
        include_paths: IncludePathIter,
        link_paths: LinkPathIter,
        libraries: LibraryIter,
    ) -> Result<()>
    where
        SourcePath: AsRef<Path>,
        IncludePath: AsRef<Path>,
        LinkPath: AsRef<Path>,
        Library: AsRef<str>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
        IncludePathIter: IntoIterator<Item = IncludePath>,
        LinkPathIter: IntoIterator<Item = LinkPath>,
        LibraryIter: IntoIterator<Item = Library>,
    {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                build_cuda_ext_unix(
                    build,
                    use_python,
                    cargo_commands,
                    sources,
                    include_paths,
                    link_paths,
                    libraries,
                )?;
            } else if #[cfg(target_os = "windows")] {
                unimplemented!();
            } else {
                bail!("Unsupported OS")l
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn build_cuda_ext_unix<
        SourcePath,
        IncludePath,
        LinkPath,
        Library,
        SourcePathIter,
        IncludePathIter,
        LinkPathIter,
        LibraryIter,
    >(
        build: &mut cc::Build,
        use_python: bool,
        mut cargo_commands: Option<&mut Vec<String>>,
        sources: SourcePathIter,
        include_paths: IncludePathIter,
        link_paths: LinkPathIter,
        libraries: LibraryIter,
    ) -> Result<()>
    where
        SourcePath: AsRef<Path>,
        IncludePath: AsRef<Path>,
        LinkPath: AsRef<Path>,
        Library: AsRef<str>,
        SourcePathIter: IntoIterator<Item = SourcePath>,
        IncludePathIter: IntoIterator<Item = IncludePath>,
        LinkPathIter: IntoIterator<Item = LinkPath>,
        LibraryIter: IntoIterator<Item = Library>,
    {
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
            .includes(include_paths)
            .flag("-std=c++14")
            .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi_flag))
            .files(sources);

        // specify CUDA architecture flags
        cuda_arches.iter().for_each(|arch| {
            build.flag(&arch.nvcc_flag());
        });

        // link libtorch
        libtorch.link_paths(USE_CUDA_API)?.for_each(|path| {
            add_link_path_unix(build, &path, &mut cargo_commands);
        });
        libtorch
            .libraries(USE_CUDA_API, use_python)?
            .for_each(|library| {
                add_library_unix(build, library, &mut cargo_commands);
            });

        // link user-specified libraries
        libraries.into_iter().for_each(|library| {
            add_library_unix(build, library.as_ref(), &mut cargo_commands);
        });
        link_paths.into_iter().for_each(|path| {
            add_link_path_unix(build, path.as_ref(), &mut cargo_commands);
        });

        // link python
        if use_python {
            link_python_libs_unix(build, &mut cargo_commands)?;
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn link_python_libs_unix(
        build: &mut cc::Build,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) -> Result<()> {
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
                    add_link_path_unix(build, Path::new(path), cargo_commands);
                }
                Some("-l") => {
                    let library = &flag[2..];
                    add_library_unix(build, library, cargo_commands);
                }
                _ => {
                    warn!("ignore `python3-config` flag {}", flag);
                }
            });

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn add_link_path_unix(
        build: &mut cc::Build,
        path: &Path,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) {
        build
            .flag("-Xlinker")
            .flag(&format!("-Wl,-rpath={}", path.display()));
        if let Some(cargo_commands) = cargo_commands {
            cargo_commands.push(format!("cargo:rustc-link-search=native={}", path.display()));
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn add_library_unix(
        build: &mut cc::Build,
        library: &str,
        cargo_commands: &mut Option<&mut Vec<String>>,
    ) {
        build.flag(&format!("-l{}", library));
        if let Some(cargo_commands) = cargo_commands {
            cargo_commands.push(format!("cargo:rustc-link-lib={}", library));
        }
    }
}
