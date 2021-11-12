#![cfg(feature = "download-libtorch")]

use anyhow::{anyhow, Result};
use std::{
    env, fs, io,
    io::prelude::*,
    path::{Path, PathBuf},
};
use crate::globals::TORCH_VERSION;
use crate::globals::LIBTORCH_URL;

pub fn download_libtorch() -> Result<PathBuf> {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR is not set");
    let libtorch_dir = PathBuf::from(out_dir).join("libtorch");
    fs::create_dir_all(&libtorch_dir)?;
    let path = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
    download(&*LIBTORCH_URL, &path)?;
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
