cmake_minimum_required(VERSION 3.18)
project(xio_sdma_ep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)
find_package(hsa-runtime64 REQUIRED)

# pybind11
find_package(pybind11 REQUIRED)

# hsakmt
find_library(HSAKMT_LIB hsakmt PATHS /opt/rocm/lib REQUIRED)

set(SDMA_EP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/endpoints/sdma-ep")

pybind11_add_module(sdma_ep
  sdma_ep_bindings.cpp
  anvil_standalone.cpp
)

target_include_directories(sdma_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SDMA_EP_DIR}
  /opt/rocm/include
  /opt/rocm/include/hsakmt
)

target_link_libraries(sdma_ep PRIVATE
  hip::host
  hsa-runtime64::hsa-runtime64
  ${HSAKMT_LIB}
)

# Install into xio/ package directory
install(TARGETS sdma_ep DESTINATION xio)
install(FILES xio/__init__.py DESTINATION xio)
