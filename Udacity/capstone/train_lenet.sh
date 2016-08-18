#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=capstone/mnist/lenet_solver.prototxt $@
