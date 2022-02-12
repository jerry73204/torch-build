//! CUDA related types and functions.

use crate::env::TORCH_CUDA_ARCH_LIST;
use anyhow::{anyhow, Error, Result};
use indexmap::IndexSet;
use itertools::Itertools as _;
use once_cell::sync::{Lazy, OnceCell};
use std::{cmp, collections::HashMap, str::FromStr};

static CUDA_ARCH_ALIASES: Lazy<HashMap<String, Vec<CudaArch>>> = Lazy::new(|| {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/CUDA_ARCH_ALIASES"))
        .lines()
        .map(|line| {
            let mut tokens = line.split('\t');
            let name = tokens.next().unwrap();
            let list = tokens.next().unwrap();
            assert!(tokens.next().is_none());
            let arches: Vec<_> = list
                .split(';')
                .map(CudaArch::from_str)
                .try_collect()
                .unwrap();
            (name.to_string(), arches)
        })
        .collect()
});

/// The CUDA architecture version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaArch {
    pub major: u32,
    pub minor: u32,
    pub with_ptx: bool,
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

    /// Parse the `;` seperated list of architecture numbers.
    ///
    /// For example, `3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6`.
    pub fn parse_list(text: &str) -> Result<Vec<Self>> {
        let arches: Vec<_> = text
            .split(';')
            .flat_map(|token| {
                if let Some(list) = CUDA_ARCH_ALIASES.get(token) {
                    list.iter().map(|arch| Ok(arch.clone())).collect()
                } else {
                    vec![token.parse()]
                }
            })
            .try_collect()?;

        Ok(arches)
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

/// Generate compatible architecture for the host system.
pub fn cuda_arches() -> Result<&'static [CudaArch]> {
    static MAX_CUDA_ARCH: Lazy<(u32, u32)> = Lazy::new(|| {
        let max = TORCH_CUDA_ARCH_LIST.iter().max().unwrap();
        (max.major, max.minor)
    });

    static ARCHES: OnceCell<Vec<CudaArch>> = OnceCell::new();

    let arches = ARCHES.get_or_try_init(|| -> Result<_> {
        use rustacuda::{
            device::{Device, DeviceAttribute::*},
            CudaFlags,
        };

        rustacuda::init(CudaFlags::empty())?;

        let host_arches: IndexSet<_> = Device::devices()?
            .map(|device| -> Result<_> {
                let device = device?;
                let major = device.get_attribute(ComputeCapabilityMajor)? as u32;
                let minor = device.get_attribute(ComputeCapabilityMinor)? as u32;
                let version = (major, minor);
                Ok(cmp::min(version, *MAX_CUDA_ARCH))
            })
            .try_collect()?;
        let mut host_arches: Vec<_> = host_arches
            .into_iter()
            .map(|(major, minor)| CudaArch {
                major,
                minor,
                with_ptx: false,
            })
            .collect();
        host_arches.sort();
        host_arches.last_mut().unwrap().with_ptx = true;

        Ok(host_arches)
    })?;

    Ok(arches.as_ref())
}
