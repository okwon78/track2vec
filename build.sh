#!/bin/bash

WORKDIR="$( cd "$( dirname "$0" )" && pwd )"

function build_track2vec {
    rm -rf $WORKDIR/build
    mkdir $WORKDIR/build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make
    cp $WORKDIR/build/track2vec $WORKDIR
}


git pull
build_track2vec

chmod +x $WORKDIR/track2vec

PREVIOUS_OUTPUT=$HOME/output

if [ -d "$PREVIOUS_OUTPUT" ]; then
    cp -r $PREVIOUS_OUTPUT $WORKDIR/output    
fi

