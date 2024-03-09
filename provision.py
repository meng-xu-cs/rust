#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import pathlib
    import shutil
    import subprocess

    # paths
    path_base = pathlib.Path(__file__).parent.resolve()
    path_x = path_base.joinpath("x")
    path_config = path_base.joinpath("config.toml")

    path_src_std = path_base.joinpath("library", "std")

    path_build = path_base.joinpath("build", "host")
    path_build_stage1_base = path_build.joinpath("stage1", "bin")
    path_build_stage1_tool = path_build.joinpath("stage1-tools-bin")

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
    if not path_build_stage1_base.exists():
        raise RuntimeError("{} does not exist".format(path_build_stage1_base))
    if not path_build_stage1_tool.exists():
        raise RuntimeError("{} does not exist".format(path_build_stage1_tool))

    # merge the binaries in the tools
    for item in os.listdir(path_build_stage1_tool):
        dst = path_build_stage1_base.joinpath(item)
        if not dst.exists():
            shutil.copy2(path_build_stage1_tool.joinpath(item), dst)

    # dump the information
    print("Toolchain: {}".format(path_build_stage1_tool))
    print("Stdlib: {}".format(path_src_std))
