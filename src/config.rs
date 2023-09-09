use anyhow::{anyhow, Error, Result};
use once_cell::sync::Lazy;
use serde::{de::Error as _, Deserialize, Deserializer};
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub torch_version: String,
    pub torch_cuda_arch_list: HashSet<CudaArch>,
    pub cuda_arch_aliases: HashMap<String, Vec<CudaArch>>,
}

/// The CUDA architecture version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaArch {
    pub major: u32,
    pub minor: u32,
    pub with_ptx: bool,
}

impl<'de> Deserialize<'de> for CudaArch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let ver: Self = text.parse().map_err(|err| {
            D::Error::custom(format!("'{text}' is not a valid CUDA arch version: {err}"))
        })?;
        Ok(ver)
    }
}

impl CudaArch {
    /// Generate the nvcc flag for this architecture.
    ///
    /// It generates the flag if version is `X.Y`.
    /// - If `with_ptx=true`, `-gencode=arch=sm_X,code=sm_Y`
    /// - If `with_ptx=true`, `-gencode=arch=compute_X,code=sm_Y`
    pub fn nvcc_flag(&self) -> String {
        let Self {
            major,
            minor,
            with_ptx,
        } = *self;
        let number = format!("{}{}", major, minor);
        let code_kind = if with_ptx { "compute" } else { "sm" };

        format!(
            "-gencode=arch=compute_{},code={}_{}",
            number, code_kind, number
        )
    }
}

impl FromStr for CudaArch {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        static REGEX_CUDA_ARCH: Lazy<regex::Regex> =
            Lazy::new(|| regex::Regex::new(r"^(\d+)\.(\d+)(\+PTX)?$").unwrap());

        let cap = REGEX_CUDA_ARCH
            .captures(text)
            .ok_or_else(|| anyhow!(r#"invalid CUDA arch "{}""#, text))?;

        let major = cap.get(1).unwrap().as_str().parse().unwrap();
        let minor = cap.get(2).unwrap().as_str().parse().unwrap();
        let with_ptx = cap.get(3).is_some();

        Ok(Self {
            major,
            minor,
            with_ptx,
        })
    }
}
