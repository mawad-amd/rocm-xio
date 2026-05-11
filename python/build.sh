#!/bin/bash
# Build xio.sdma_ep as part of the full rocm-xio cmake build.
#
# Usage: bash python/build.sh [gfx942|gfx950]
# Run from the rocm-xio repo root.

set -e

ARCH="${1:-gfx942}"

echo "Building rocm-xio + Python bindings for ${ARCH}..."

# Install pybind11 if not present
python3 -c "import pybind11" 2>/dev/null || pip install pybind11

mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DOFFLOAD_ARCH="${ARCH}" \
  -DBUILD_CLIENTS=OFF \
  -DBUILD_TESTING=OFF \
  2>&1

# Build just the sdma_ep pybind11 module (and its dependency librocm-xio.so)
make -j$(nproc) sdma_ep 2>&1

echo ""
echo "Built successfully. To install:"
echo "  export PYTHONPATH=$(pwd)/python:\$PYTHONPATH"
echo "  python3 -c 'from xio import sdma_ep; print(sdma_ep.SDMA_QUEUE_SIZE)'"
