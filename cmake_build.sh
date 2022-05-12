#!/bin/bash

WORKDIR=$HOME

CMAKE_VER="v3.18.3"

function build_cmake {
    git clone https://github.com/Kitware/CMake.git $WORKDIR/CMake
    cd $WORKDIR/CMake
    git checkout tags/$CMAKE_VER
    $WORKDIR/CMake/bootstrap --prefix=/usr/local
    make -j$(nproc) && make install
}

build_cmake