# Copyright 2020 Johannes Demel
#
# This file is part of PolarCodes
#
# SPDX-License-Identifier: GPL-3.0-or-later
#



add_executable (multibench multiply_benchmark.cpp)

target_link_libraries(multibench benchmark::benchmark fmt::fmt Volk::volk)
# target_compile_options(multibench PUBLIC "-mavx2")
# target_compile_options(multibench PUBLIC "-march=znver3"  "-mavx2")
target_compile_options(multibench PUBLIC "-fopenmp-simd" "-ftree-vectorize" "-fopt-info-vec-optimized" "-m64" "-mmmx" "-msse" "-msse2" "-msse3" "-mssse3" "-msse4.1" "-msse4.2" "-mpopcnt" "-mavx" "-mfma" "-mavx2")


# target_include_directories(mapbench
#     PRIVATE $<BUILD_INTERFACE:${SYMBOLMAPPING_INCLUDE_DIRS}>
#     PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
#     PUBLIC $<INSTALL_INTERFACE:include>
# )
