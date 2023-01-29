# Copyright 2021 Johannes Demel
#
# This file is part of multiply_benchmark
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Project setup
########################################################################
cmake_minimum_required(VERSION 3.22)
project ("multiply_benchmark" VERSION 1.0 LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

#select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release")
   message(STATUS "Build type not specified: defaulting to release.")
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#make sure our local CMake Modules path comes first
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# include_directories(${CMAKE_SOURCE_DIR}/include)

########################################################################
# Specify dependency packages
########################################################################
find_package(fmt REQUIRED)
find_package(Volk REQUIRED 3.0)
find_package(benchmark REQUIRED 1.5.3)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-O3)
else()
    add_definitions(-Og)
endif()


########################################################################
# Add subdirectories with actual code
########################################################################
add_subdirectory(src)


########################################################################
# Install cmake search helper for this library
########################################################################
# if(NOT CMAKE_MODULES_DIR)
#   set(CMAKE_MODULES_DIR lib/cmake)
# endif(NOT CMAKE_MODULES_DIR)

# install(FILES cmake/Modules/SDRBenchmarkConfig.cmake
#     DESTINATION ${CMAKE_MODULES_DIR}/SDRBenchmark
# )

message(STATUS "CMake Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Using install libdir: ${INSTALL_FULL_LIBDIR}")
