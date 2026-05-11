# Python bindings for rocm-xio SDMA endpoint.
#
# This is added to the top-level build via add_subdirectory(python).
# Requires BUILD_SHARED_LIBS=ON and pybind11.
#
# Usage (standalone):
#   mkdir build && cd build
#   cmake -DBUILD_SHARED_LIBS=ON -DOFFLOAD_ARCH=gfx942 ..
#   make -j sdma_ep
#
# Usage (as part of top-level rocm-xio build):
#   add_subdirectory(python)  # in top-level CMakeLists.txt

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
  )
  if(PYBIND11_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
  endif()
endif()

# Must compile as HIP — sdma-ep.h has __device__ functions with HIP builtins
set_source_files_properties(sdma_ep_bindings.cpp PROPERTIES LANGUAGE HIP)

pybind11_add_module(sdma_ep sdma_ep_bindings.cpp)

# Need -fgpu-rdc to link against rocm-xio (built with -fgpu-rdc)
target_link_options(sdma_ep PRIVATE -fgpu-rdc)

target_include_directories(sdma_ep PRIVATE
  ${CMAKE_SOURCE_DIR}/src/include
  ${CMAKE_SOURCE_DIR}/src/endpoints/sdma-ep
  ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(sdma_ep PRIVATE
  rocm-xio
  hip::host
)

if(XIO_SDMA_OSS7)
  target_compile_definitions(sdma_ep PRIVATE XIO_SDMA_OSS7=1)
endif()

# Install the module into the xio package
install(TARGETS sdma_ep
  LIBRARY DESTINATION ${Python3_SITEARCH}/xio
)
install(FILES xio/__init__.py
  DESTINATION ${Python3_SITEARCH}/xio
)
