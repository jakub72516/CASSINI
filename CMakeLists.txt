cmake_minimum_required(VERSION 3.27)

project(Cassini
  VERSION 0.1.0
  DESCRIPTION "Real-time N-body relativistic solar system simulator"
  LANGUAGES CXX)

# ---- Global policy / standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Feature toggles ----
option(CASSINI_ENABLE_CUDA      "Enable CUDA gravity backend"        OFF)
option(CASSINI_ENABLE_OPENMP    "Enable OpenMP CPU parallelism"      ON)
option(CASSINI_ENABLE_HDF5      "Enable HDF5 / SPICE export"         OFF)
option(CASSINI_ENABLE_TELEMETRY "Enable gRPC/WebSocket telemetry"    OFF)
option(CASSINI_ENABLE_RENDER    "Enable OpenGL renderer"             OFF)
option(CASSINI_BUILD_TESTS      "Build test suite"                   ON)
option(CASSINI_BUILD_BENCHMARKS "Build benchmark suite"              ON)
option(CASSINI_ENABLE_SANITIZERS "Enable ASan/UBSan (GCC/Clang)"     OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerWarnings)
include(Sanitizers)

enable_project_warnings()

if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ---- Core header-only library ----
add_library(cassini INTERFACE)
add_library(cassini::cassini ALIAS cassini)
target_include_directories(cassini INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cassini INTERFACE cxx_std_20)

# ---- OpenMP ----
if(CASSINI_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cassini INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(cassini INTERFACE CASSINI_HAS_OPENMP=1)
  endif()
endif()

# ---- CUDA backend (separate static lib so headers stay host-only) ----
if(CASSINI_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_library(cassini_cuda STATIC
    src/gravity/cuda/cuda_gravity.cu)
  target_include_directories(cassini_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(cassini_cuda PUBLIC cassini CUDA::cudart)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(cassini_cuda PROPERTIES CUDA_ARCHITECTURES "86")
  endif()
  add_library(cassini::cuda ALIAS cassini_cuda)
endif()

# ---- HDF5 ----
if(CASSINI_ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
  add_library(cassini_hdf5 STATIC src/data/hdf5_export.cpp)
  target_include_directories(cassini_hdf5 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(cassini_hdf5 PUBLIC cassini HDF5::HDF5)
  add_library(cassini::hdf5 ALIAS cassini_hdf5)
endif()

# ---- App ----
add_executable(cassini_cli src/app/main.cpp)
target_link_libraries(cassini_cli PRIVATE cassini)
if(CASSINI_ENABLE_CUDA)
  target_link_libraries(cassini_cli PRIVATE cassini_cuda)
  target_compile_definitions(cassini_cli PRIVATE CASSINI_HAS_CUDA=1)
endif()

# ---- Tests / benchmarks ----
if(CASSINI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(CASSINI_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
