#![cfg(feature = "download-libtorch")]

use crate::env::{TORCH_CUDA_VERSION, TORCH_VERSION};
use anyhow::{anyhow, Result};
use cfg_if::cfg_if;
use once_cell::sync::OnceCell;
use std::{
    fs, io,
    io::prelude::*,
    path::{Path, PathBuf},
};

pub(crate) fn download_libtorch() -> Result<PathBuf> {
    let libtorch_dir = PathBuf::from(crate::env::OUT_DIR).join("libtorch");
    fs::create_dir_all(&libtorch_dir)?;
    let path = libtorch_dir.join(format!("v{}.zip", *TORCH_VERSION));
    download(libtorch_url()?, &path)?;
    extract(&path, &libtorch_dir)?;
    let libtorch_dir = libtorch_dir.join("libtorch");
    Ok(libtorch_dir)
}

fn download(source_url: &str, target_file: impl AsRef<Path>) -> Result<()> {
    let mut reader = ureq::get(source_url).call()?.into_reader();
    let mut writer = io::BufWriter::new(fs::File::create(&target_file)?);
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn torch_device_literal() -> &'static str {
    static LITERAL: OnceCell<String> = OnceCell::new();

    LITERAL.get_or_init(|| {
        TORCH_CUDA_VERSION
            .as_ref()
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
    })
}

/// Generates the libtorch download URL according to host operating system.
pub fn libtorch_url() -> Result<&'static str> {
    static URL: OnceCell<String> = OnceCell::new();

    URL.get_or_try_init(|| -> Result<_> {
        let device = torch_device_literal();

        let url = {
            cfg_if! {
                if #[cfg(target_os = "linux")] {
                    let use_cxx11_abi = crate::probe::check_cxx11_abi();

                    // XXX: the indentation prevents rustfmt to crash
                    format!(
                        "https://download.pytorch.org/libtorch/\
                         {}/libtorch{}-abi-shared-with-deps-{}%2B{}.zip",
                        device,
                        if use_cxx11_abi { "-cxx11" } else { "" },
                        *TORCH_VERSION,
                        device
                    )

                } else if #[cfg(target_os = "macos")] {
                    format!(
                        "https://download.pytorch.org/libtorch/\
                         cpu/libtorch-macos-{}.zip",
                        TORCH_VERSION
                    )
                } else if #[cfg(target_os = "windows")] {
                    format!(
                        "https://download.pytorch.org/libtorch/\
                         {}/libtorch-win-shared-with-deps-{}%2B{}.zip",
                        device, TORCH_VERSION, device
                    )
                } else {
                    bail!("Unsupported OS")
                }
            }
        };

        Ok(url)
    })
    .map(|url| url.as_str())
}

fn extract(filename: impl AsRef<Path>, outpath: impl AsRef<Path>) -> Result<()> {
    let buf = io::BufReader::new(fs::File::open(&filename)?);
    let mut archive = zip::ZipArchive::new(buf)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let path = file.enclosed_name().ok_or_else(|| {
            anyhow!(
                "unable to extract zip file due to unenclosed name '{}'",
                file.name()
            )
        })?;
        let outpath = outpath.as_ref().join(path);

        if file.is_file() {
            eprintln!(
                r#"File {} extracted to "{}" ({} bytes)"#,
                i,
                outpath.display(),
                file.size()
            );
            let mut outfile = io::BufWriter::new(fs::File::create(&outpath)?);
            io::copy(&mut file, &mut outfile)?;
        } else {
            fs::create_dir_all(path)?;
        }
    }
    Ok(())
}
