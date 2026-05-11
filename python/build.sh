#!/bin/bash
# Build xio.sdma_ep pybind11 module directly without cmake.
# Run inside a ROCm container with pybind11 installed.
#
# Usage: cd python && bash build.sh
# Output: xio/sdma_ep.cpython-*.so

set -e

SDMA_EP_DIR="../src/endpoints/sdma-ep"
ROCM="/opt/rocm"

PYTHON_INCLUDES=$(python3 -m pybind11 --includes)
PYTHON_SUFFIX=$(python3-config --extension-suffix)

g++ -O2 -shared -fPIC -std=c++17 \
  ${PYTHON_INCLUDES} \
  -I. \
  -I${SDMA_EP_DIR} \
  -I${ROCM}/include \
  -I${ROCM}/include/hsakmt \
  sdma_ep_bindings.cpp \
  anvil_standalone.cpp \
  -L${ROCM}/lib \
  -lamdhip64 \
  -lhsa-runtime64 \
  -lhsakmt \
  -o xio/sdma_ep${PYTHON_SUFFIX}

echo "Built xio/sdma_ep${PYTHON_SUFFIX}"
echo "Test: cd .. && python3 -c 'from xio import sdma_ep; print(sdma_ep.SDMA_QUEUE_SIZE)'"
