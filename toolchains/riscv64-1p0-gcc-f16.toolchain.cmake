#========================================================
# Copyright (c) RVVPL and Lobachevsky State University of 
# Nizhny Novgorod and its affiliates. All rights reserved.
# 
# Copyright 2025-2026 The RVVMF Authors (Evgeny Kozinov)
#
# Distributed under the BSD 4-Clause License
# (See file LICENSE in the root directory of this 
# source tree)
#========================================================
#
#********************************************************
#                                                       *
#   File: riscv64-1p0-gcc-f16.toolchain.cmake           *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)

set(CMAKE_CXX_FLAGS_INIT "-march=rv64gcv_zvfh -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT   "-march=rv64gcv_zvfh -mabi=lp64d")