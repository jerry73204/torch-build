# torch-build

Utilities to link libtorch FFI interface.

## Usage

Add `cc` and `torch-build` build dependencies in your `Cargo.toml`.

```toml
[build-dependencies]
anyhow = "1.0.45"
cc = "1.0.72"
torch-build = "X.Y.Z"
```

The `build.rs` snipplet compiles C++/CUDA source files and links to libtorch.

```rust
use anyhow::Result;

// Provides C++/CUDA source files
const CPP_SOURCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/nms_cpu.cpp");
const CUDA_SOURCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/nms_cuda.cu");

fn main() -> Result<()> {
    // The container stores cargo-build commands
    let mut cargo_commands = vec![];

    // Compile C++ files
    let mut build = cc::Build::new();
    torch_build::build_cpp(
        &mut build,
        true,
        false,
        Some(&mut cargo_commands),
        [CPP_SOURCE],
    )?;
    build.try_compile("nms_cpu")?;

    // Compile CUDA files
    let mut build = cc::Build::new();
    torch_build::build_cuda(&mut build, false, Some(&mut cargo_commands), [CUDA_SOURCE])?;
    build.try_compile("nms_cuda")?;

    // Re-compile if C++/CUDA source files were modified
    println!("cargo:rerun-if-changed={}", CPP_SOURCE);
    println!("cargo:rerun-if-changed={}", CUDA_SOURCE);
    cargo_commands.iter().for_each(|command| {
        println!("{}", command);
    });

    Ok(())
}
```

## License

MIT license. See [license file](LICENSE).
