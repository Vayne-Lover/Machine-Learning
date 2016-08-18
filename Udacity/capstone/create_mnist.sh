#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

CAPSTONE=capstone/mnist
DATA=data/mnist
BUILD=build/capstone/mnist

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $CAPSTONE/mnist_train_${BACKEND}
rm -rf $CAPSTONE/mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $CAPSTONE/mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $CAPSTONE/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
