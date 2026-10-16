use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
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
}
