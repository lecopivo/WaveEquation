#!/usr/bin/bash

src_path=`readlink -f ..`

if [ ! -f ${src_path}/.dir-locals.el ]; then
    echo "Creating new dir-locals.el!"
    echo "((c++-mode (cmake-ide-build-dir . \"${src_path}/build/debug\")))" > ../.dir-locals.el
fi

# debug build
rm -f debug/ -R && mkdir -p debug/ && cd debug/
cmake -DCMAKE_BUILD_TYPE="Debug" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      ../../

cd ../

# release build
rm -f release/ -R && mkdir -p release/ && cd release/
cmake -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      ../../

cd ../
