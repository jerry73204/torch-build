use anyhow::Result;
use std::{env, fs, path::PathBuf, str};

fn main() -> Result<()> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let use_cxx11_abi = {
        let bytes = cc::Build::new()
            .cpp(true)
            .warnings(false)
            .file("csrc/test_cxx11_abi.cpp")
            .expand();
        str::from_utf8(&bytes).unwrap().contains("YES")
    };

    if use_cxx11_abi {
        let path = out_dir.join("use_cxx11_abi");
        fs::File::create(path)?;
    }

    Ok(())
}
