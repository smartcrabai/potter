use std::{env, io, path::PathBuf};

fn main() -> Result<(), io::Error> {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "CARGO_MANIFEST_DIR is not set")
        })?);

    let skin_native = manifest_dir.join("src/geom/modifiers/skin_native");
    let bullet = skin_native.join("bullet/LinearMath");
    cc::Build::new()
        .cpp(true)
        .std("c++11")
        .define("BT_USE_DOUBLE_PRECISION", None)
        .include(&bullet)
        .file(skin_native.join("skin_hull.cpp"))
        .file(bullet.join("btConvexHullComputer.cpp"))
        .file(bullet.join("btAlignedAllocator.cpp"))
        .compile("potter_skin_hull");
    println!("cargo:rerun-if-changed=src/geom/modifiers/skin_native/skin_hull.cpp");
    println!("cargo:rerun-if-changed=src/geom/modifiers/skin_native/bullet/source-pin.txt");
    for source in [
        "btConvexHullComputer.cpp",
        "btConvexHullComputer.h",
        "btAlignedAllocator.cpp",
        "btAlignedAllocator.h",
        "btAlignedObjectArray.h",
        "btMinMax.h",
        "btScalar.h",
        "btVector3.h",
    ] {
        println!("cargo:rerun-if-changed={}", bullet.join(source).display());
    }

    let eigen_dir = manifest_dir.join("vendor/eigen");
    if !eigen_dir.join("Eigen/LU").is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Eigen submodule is missing; run `git submodule update --init --recursive`",
        ));
    }
    let target = env::var("TARGET").map_err(|error| io::Error::other(error.to_string()))?;
    let mut eigen = cc::Build::new();
    eigen
        .cpp(true)
        .std("c++14")
        .file(manifest_dir.join("native/blender_inverse_m4.cc"))
        .include(&eigen_dir);
    eigen.opt_level(2);
    if target.ends_with("-apple-darwin") {
        eigen
            .flag("-mdynamic-no-pic")
            .flag("-pipe")
            .flag("-funsigned-char")
            .flag("-fno-strict-aliasing")
            .flag("-ffp-contract=off");
        if target.starts_with("x86_64-") {
            eigen
                .flag("-msse")
                .flag("-msse2")
                .flag("-msse3")
                .flag("-mssse3");
        }
    } else if target.ends_with("-unknown-linux-gnu") {
        eigen
            .flag("-fPIC")
            .flag("-pipe")
            .flag("-funsigned-char")
            .flag("-fno-strict-aliasing")
            .flag("-ffp-contract=off");
    }
    eigen.compile("potter_blender_inverse");
    println!("cargo:rerun-if-changed=native/blender_inverse_m4.cc");
    println!("cargo:rerun-if-changed=vendor/eigen/Eigen/LU");
    println!("cargo:rerun-if-changed=vendor/eigen/Eigen/src/LU/InverseImpl.h");
    Ok(())
}
