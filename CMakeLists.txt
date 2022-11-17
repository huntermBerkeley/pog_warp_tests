cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(pog_tests VERSION 0.0.1 LANGUAGES CXX CUDA)



###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)


find_package(CUDAToolkit 11.4 REQUIRED)

#package manager
include(cmake/CPM.cmake)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)

set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/downloaded_libraries")

## Package Target

CPMAddPackage(
  NAME poggers
  GITHUB_REPOSITORY huntermBerkeley/poggers
  GIT_TAG origin/main
  VERSION v1.0
)


CPMAddPackage(
  NAME warpcore
  GITHUB_REPOSITORY sleeepyjack/warpcore
  GIT_TAG origin/master

)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 

#needed for warpcore
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")


ConfigureExecutable(test "${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu" "${CMAKE_BINARY_DIR}")


target_link_libraries(test PRIVATE poggers)
target_link_libraries(test PRIVATE warpcore)



