cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(example VERSION 0.0.1 LANGUAGES CXX)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

## Setup SYCL
### Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")
find_package(hipSYCL CONFIG REQUIRED)
find_package(KissRng CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(example "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp")
target_link_libraries(example PRIVATE KissRng::kiss OpenMP::OpenMP_CXX)
