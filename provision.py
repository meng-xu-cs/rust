#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import pathlib
    import subprocess

    # paths
    path_base = pathlib.Path(__file__).parent.resolve()
    path_x = path_base.joinpath("x")
    path_config = path_base.joinpath("config.toml")

    path_src_std = path_base.joinpath("library", "std")

    path_build = path_base.joinpath("build", "host")
    path_build_stage0_tool_cargo = path_build.joinpath("stage0", "bin", "cargo")
    path_build_stage1_bin = path_build.joinpath("stage1", "bin")
    path_build_stage1_tool_cargo = path_build.joinpath("stage1-tools-bin", "cargo")

    # setup
    if not path_config.exists():
        subprocess.check_call([path_x, "setup", "compiler"])

        # extend the config file
        content = []
        with open(path_config) as f_reader:
            for line in f_reader:
                content.append(line.rstrip())

        content.extend(
            [
                "[build]",
                "extended = true",
            ]
        )

        with open(path_config, "w") as f_writer:
            for line in content:
                f_writer.write("{}\n".format(line))

    # build
    subprocess.check_call([path_x, "build"])
    if not path_build_stage1_bin.exists():
        raise RuntimeError("{} does not exist".format(path_build_stage1_bin))
    if not path_build_stage1_tool_cargo.exists():
        raise RuntimeError("{} does not exist".format(path_build_stage1_tool_cargo))

    # wrap the cargo tool with a script
    dst = path_build_stage1_bin.joinpath("cargo")
    if dst.exists():
        raise RuntimeError("{} already exists".format(dst))

    with open(dst, "w") as f:
        f.write(
            """#!/bin/bash
if [ "$1" == "check" ]; then
    exec env RUSTC_BOOTSTRAP=1 {} "$@"
else  
    exec env RUSTC_BOOTSTRAP=1 {} "$@"
fi""".format(
                path_build_stage0_tool_cargo, path_build_stage1_tool_cargo
            )
        )

    os.chmod(dst, 0o755)

    # dump the information
    print("Toolchain: {}".format(path_build_stage1_bin))
    print("Stdlib: {}".format(path_src_std))
