use std::path::Path;

pub fn print_cargo_link_search<P>(path: P)
where
    P: AsRef<Path>,
{
    let display = path.as_ref().display();
    println!("cargo:rustc-link-search=native={display}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{display}");
}

pub fn print_cargo_link_library(name: &str) {
    println!("cargo:rustc-link-lib={name}",);
}
