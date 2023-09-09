//! CUDA related types and functions.

use crate::{config::CudaArch, env::TORCH_CUDA_ARCH_LIST};
use anyhow::Result;
use indexmap::IndexSet;
use itertools::Itertools as _;
use once_cell::sync::{Lazy, OnceCell};
use std::cmp;

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
