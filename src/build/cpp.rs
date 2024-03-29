use crate::{probe_python, ProbePython};
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
    headers: Vec<PathBuf>,
    sources: Vec<PathBuf>,
}

impl CppExtension {
    pub fn new() -> Self {
        Self {
            use_cuda_api: false,
            link_python: false,
            includes: vec![],
            headers: vec![],
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

    pub fn header<P>(&mut self, path: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.headers.push(path.as_ref().to_owned());
        self
    }

    pub fn headers<P>(&mut self, paths: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self.headers
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
        self.configure_cc(&mut build)?;
        build.try_compile(name)?;
        self.link()?;
        Ok(())
    }

    /// Configure the [cc::Build] to compile C++ source code.
    pub fn configure_cc(&self, build: &mut cc::Build) -> Result<()> {
        cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "macos"))] {
                self.configure_cc_unix(build)?
            } else if #[cfg(target_os = "windows")] {
                self.configure_cc_windows(build)?
            } else {
                bail!("Unsupported OS")
            }
        }

        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn configure_cc_unix(&self, build: &mut cc::Build) -> Result<()> {
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
            configure_cc_python_libs_unix(build)?;
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn configure_cc_windows(&self, build: &mut cc::Build) -> Result<()> {
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
        use crate::build::utils::{print_cargo_link_library, print_cargo_link_search};

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
            print_cargo_link_search(path);
        });
        libtorch
            .libraries(use_cuda_api, link_python)?
            .for_each(|library| {
                print_cargo_link_library(library);
            });

        // link user-specified libraries
        link_searches.iter().for_each(|path| {
            print_cargo_link_search(path);
        });
        libraries.iter().for_each(|library| {
            print_cargo_link_library(library);
        });

        // link python
        if link_python {
            link_python_libs_unix()?;
        }

        Ok(())
    }

    pub fn configure_bindgen(&self, builder: bindgen::Builder) -> Result<bindgen::Builder> {
        let Self {
            use_cuda_api,
            link_python,
            ref includes,
            ref headers,
            ..
        } = *self;

        // Probe libtorch
        let libtorch = crate::probe::probe_libtorch()?;

        let builder = builder.clang_args(["-x", "c++"]);

        let builder = headers.iter().fold(builder, |builder, header| {
            builder.header(format!("{}", header.display()))
        });

        let builder = includes.iter().fold(builder, |builder, path| {
            builder.clang_arg(format!("-I{}", path.display()))
        });

        let builder = libtorch
            .include_paths(use_cuda_api)?
            .fold(builder, |builder, path| {
                builder.clang_arg(format!("-I{}", path.display()))
            });

        let builder = if link_python {
            let ProbePython {
                includes: python_includes,
                ..
            } = probe_python()?;

            python_includes.into_iter().fold(builder, |builder, path| {
                builder.clang_arg(format!("-I{}", path.display()))
            })
        } else {
            builder
        };

        Ok(builder)
    }
}

impl Default for CppExtension {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn configure_cc_python_libs_unix(build: &mut cc::Build) -> Result<()> {
    let ProbePython {
        includes,
        link_searches,
        libraries,
    } = probe_python()?;
    build.includes(includes);

    for path in link_searches {
        build.flag(&format!("-Wl,-rpath={}", path.display()));
    }

    for library in libraries {
        build.flag(&format!("-l{library}"));
    }

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
