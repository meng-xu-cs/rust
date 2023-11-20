#!/bin/bash -e

# Usage:
#
# -f: force rebuild after cleaning the previous build
# -d: distribution build, takes longer time
#
# In normal cases, for incremental development,
# simply run this script without any arguments.
#
# A Rust toolchain tagged `fuzz` will be created at the end
# if everything goes fine.

PREFIX=dist/fuzz
X_DIST=0

# command line flags
while test $# -gt 0; do
    case "$1" in
        -f)
            ./x clean
            rm -rf build
            rm -rf dist
            rm config.toml
            ;;
        -d)
            X_DIST=1
            ;;
        *)
            echo "invalid argument $1"
            exit 1
            ;;
    esac
    shift
done

# configure
if [ ! -f config.toml ]; then
    ./configure \
        --tools="cargo,miri,src,analysis" \
        --set install.prefix=${PREFIX} \
        --set install.sysconfdir=sysconf \
        --disable-docs \
        --disable-compiler-docs
fi

# build, install, and (optionally) distribute
./x install

if [ "$X_DIST" -eq "1" ]; then
    ./x dist
fi

# setup the toolchain
rustup toolchain link fuzz ${PREFIX}
