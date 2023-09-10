use anyhow::{ensure, Context as _, Result};
use cfg_if::cfg_if;
use log::warn;
use std::{
    path::{Path, PathBuf},
    process::Command,
    str,
};

#[derive(Debug, Clone)]
pub struct CppExtension {
    use_cuda_api: bool,
    link_python: bool,
    includes: Vec<PathBuf>,
    link_searches: Vec<PathBuf>,
    libraries: Vec<String>,
    sources: Vec<PathBuf>,
}

impl CppExtension {
    pub fn new() -> Self {
        Self {
            use_cuda_api: false,
            link_python: false,
            includes: vec![],
            sources: vec![],
            link_searches: vec![],
            libraries: vec![],
        }
    }

    pub fn use_cuda_api(&mut self, enabled: bool) -> &mut Self {
        self.use_cuda_api = enabled;
        self
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

    /// Compile and link C++ source code. This is a shorthand for
    /// [configure()](CppExtension::configure) and then
    /// [link()](CppExtension::link).
    pub fn build(&self, name: &str) -> Result<()> {
        let mut build = cc::Build::new();
        self.configure(&mut build)?;
        build.try_compile(name)?;
        self.link()?;
        Ok(())
    }

    /// Configure the [cc::Build] to compile C++ source code.
    pub fn configure(&self, build: &mut cc::Build) -> Result<()> {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                self.configure_unix(build)?
            } else if #[cfg(target_os = "windows")] {
                self.configure_windows(build)?
            } else {
                bail!("Unsupported OS")
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn configure_unix(&self, build: &mut cc::Build) -> Result<()> {
        let Self {
            use_cuda_api,
            link_python,
            ref sources,
            ref includes,
            ref libraries,
            ref link_searches,
            ..
        } = *self;

        let libtorch = crate::probe::probe_libtorch()?;
        let cxx11_abi_flag = if libtorch.use_cxx11_abi { "1" } else { "0" };

        build
            .cpp(true)
            .pic(true)
            .includes(libtorch.include_paths(use_cuda_api)?)
            .includes(includes)
            .flag("-std=c++14")
            .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi_flag))
            .files(sources);

        // link libtorch
        libtorch.link_paths(use_cuda_api)?.for_each(|path| {
            build.flag(&format!("-Wl,-rpath={}", path.display()));
        });
        libtorch
            .libraries(use_cuda_api, link_python)?
            .for_each(|lib| {
                build.flag(&format!("-l{lib}"));
            });

        // link user-specified libraries
        link_searches.iter().for_each(|path| {
            build.flag(&format!("-Wl,-rpath={}", path.display()));
        });
        libraries.iter().for_each(|lib| {
            build.flag(&format!("-l{lib}"));
        });

        // link python
        if link_python {
            configure_python_libs_unix(build)?;
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn configure_windows(&self, build: &mut cc::Build) -> Result<()> {
        let Self {
            use_cuda_api,
            link_python,
            ref sources,
            ref includes,
            ..
        } = *self;

        // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
        //       Not yet supported by cc=rs.
        //       https://github.com/alexcrichton/cc-rs/issues/323
        let libtorch = crate::probe::probe_libtorch()?;
        build
            .cpp(true)
            .pic(true)
            .includes(libtorch.include_paths(use_cuda_api)?)
            .includes(includes)
            .files(sources);
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
        let Self {
            use_cuda_api,
            link_python,
            ref libraries,
            ref link_searches,
            ..
        } = *self;

        let libtorch = crate::probe::probe_libtorch()?;

        // link libtorch
        libtorch.link_paths(use_cuda_api)?.for_each(|path| {
            println!("cargo:rustc-link-search=native={}", path.display());
        });
        libtorch
            .libraries(use_cuda_api, link_python)?
            .for_each(|library| {
                println!("cargo:rustc-link-lib={library}",);
            });

        // link user-specified libraries
        link_searches.iter().for_each(|path| {
            println!("cargo:rustc-link-search=native={}", path.display());
        });
        libraries.iter().for_each(|library| {
            println!("cargo:rustc-link-lib={library}",);
        });

        // link python
        if link_python {
            link_python_libs_unix()?;
        }

        Ok(())
    }
}

impl Default for CppExtension {
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
                build.flag(&format!("-Wl,-rpath={path}"));
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
                println!("cargo:rustc-link-search=native={path}");
            }
            Some("-l") => {
                let library = &flag[2..];
                println!("cargo:rustc-link-lib={library}",);
            }
            _ => {
                warn!("ignore `python3-config` flag {}", flag);
            }
        });

    Ok(())
}
