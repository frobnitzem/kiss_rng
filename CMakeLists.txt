cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(kiss_rng VERSION 0.0.1 LANGUAGES CXX)

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

###################################################################################################
# - define target   -------------------------------------------------------------------------------
add_library(kiss_rng INTERFACE)
target_compile_features(kiss_rng INTERFACE cxx_std_17)
target_include_directories(kiss_rng INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(KISS_RNG_BUILD_EXAMPLES "Configure CMake to build the example" OFF)
if(KISS_RNG_BUILD_EXAMPLES)
    add_executable(example "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp")
    target_link_libraries(example PRIVATE kiss_rng)
endif(KISS_RNG_BUILD_EXAMPLES)
