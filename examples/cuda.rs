use anyhow::Result;

fn main() -> Result<()> {
    let arches = torch_build::cuda::cuda_arches()?;
    dbg!(arches);
    Ok(())
}
